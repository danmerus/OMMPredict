# train_service.py
from __future__ import annotations

import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from db import MODEL_BUCKET, get_supabase, model_object_name
from features import (
    FEATURE_COLUMNS, FEATURE_VERSION, build_features, normalize_baseline, to_target,
)

APP_DIR = Path(__file__).resolve().parent         # .../Web-application (Flask)/app
DATA_DIR = APP_DIR / "data"                       # .../app/data
DEFAULT_BASELINE = DATA_DIR / "baseline_examples.csv"


@dataclass
class TrainResult:
    model: CatBoostClassifier
    metrics: Dict[str, float]
    confusion: np.ndarray
    version: str
    params: Dict[str, Any]          # CatBoost parameters of the final model
    X_cols: List[str]
    n_samples: int
    cv_folds: int
    feature_version: int = FEATURE_VERSION


def _from_supabase_verified(limit: int = 50000) -> pd.DataFrame:
    """Rows from the `predictions` table that have a confirmed outcome (`actual`)."""
    sb = get_supabase()
    res = (
        sb.table("predictions")
          .select("*")
          .not_.is_("actual", "null")
          .limit(limit)
          .execute()
    )
    df = pd.DataFrame(res.data or [])
    if df.empty:
        return df
    df["target"] = to_target(df["actual"])
    return df


def _from_local_baseline(rel_or_abs: str | None = None) -> pd.DataFrame:
    """Baseline examples from the research dataset, converted to canonical units."""
    if rel_or_abs:
        p = Path(rel_or_abs)
        p = p if p.is_absolute() else (APP_DIR / rel_or_abs)
    else:
        p = DEFAULT_BASELINE
    p = p.resolve()

    st.caption(f"📁 Поиск baseline по пути: `{p}`")
    if not p.exists():
        st.warning(f"Файл базовых примеров не найден: {p}")
        return pd.DataFrame()

    try:
        if p.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p, encoding="utf-8-sig")
    except Exception as e:
        st.warning(f"Не удалось прочитать {p}: {e}")
        return pd.DataFrame()

    if df.empty:
        st.warning(f"Файл пуст: {p}")
        return df

    st.caption(f"✅ Загружено baseline строк: {len(df)}")
    return normalize_baseline(df)


def _to_xy(df: pd.DataFrame) -> pd.DataFrame:
    out = build_features(df)
    out["target"] = df["target"].values
    return out


def assemble_training_data(include_local: bool, local_path: Optional[str]) -> pd.DataFrame:
    """Labelled rows from the database + baseline examples, as model features + `target`.

    Row counts per source are in `df.attrs["counts"]`.
    """
    parts, counts = [], {}

    df_sb = _from_supabase_verified()
    counts["db"] = len(df_sb)
    if not df_sb.empty:
        parts.append(_to_xy(df_sb))

    if include_local:
        df_base = _from_local_baseline(local_path or "")
        counts["baseline"] = len(df_base)
        if not df_base.empty:
            parts.append(_to_xy(df_base))

    if not parts:
        df = pd.DataFrame(columns=FEATURE_COLUMNS + ["target"])
        df.attrs["counts"] = counts
        return df

    df = pd.concat(parts, ignore_index=True)
    n_all = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target"])
    counts["dropped_incomplete"] = n_all - len(df)
    n_complete = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    counts["dropped_duplicates"] = n_complete - len(df)
    df["target"] = df["target"].astype(int)
    df.attrs["counts"] = counts
    return df


def _pos_weight(y: pd.Series) -> float:
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return float(neg / pos) if pos > 0 else 1.0


def train_catboost(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    iterations: int = 150,
    learning_rate: float = 0.03,
    depth: int = 3,
    l2_leaf_reg: float = 3.0,
) -> TrainResult:
    """Cross-validate on all rows (metrics come from out-of-fold predictions),
    then fit the final model on all rows."""
    X = df[FEATURE_COLUMNS].astype(float)
    y = df["target"].astype(int)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if min(pos, neg) < n_splits:
        raise ValueError(
            f"Для {n_splits}-кратной кросс-валидации нужно минимум {n_splits} примеров каждого класса "
            f"(сейчас: высокий риск = {pos}, низкий = {neg})."
        )

    base = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_state,
        verbose=False,
    )

    oof = np.zeros(len(y))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, te_idx in folds.split(X, y):
        y_tr = y.iloc[tr_idx]
        m = CatBoostClassifier(**base, scale_pos_weight=_pos_weight(y_tr))
        m.fit(X.iloc[tr_idx], y_tr)
        oof[te_idx] = m.predict_proba(X.iloc[te_idx])[:, 1]

    y_hat = (oof >= 0.5).astype(int)
    metrics = dict(
        accuracy=float(accuracy_score(y, y_hat)),
        precision=float(precision_score(y, y_hat, zero_division=0)),
        recall=float(recall_score(y, y_hat, zero_division=0)),
        f1=float(f1_score(y, y_hat, zero_division=0)),
        auc=float(roc_auc_score(y, oof)),
        pos=pos,
        neg=neg,
    )
    cm = confusion_matrix(y, y_hat, labels=[0, 1])

    params = dict(base, scale_pos_weight=_pos_weight(y))
    model = CatBoostClassifier(**params)
    model.fit(X, y)

    # Random suffix keeps versions (and their storage file names) unique.
    version = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    return TrainResult(
        model=model,
        metrics=metrics,
        confusion=cm,
        version=version,
        params=params,
        X_cols=FEATURE_COLUMNS,
        n_samples=len(df),
        cv_folds=n_splits,
    )


def save_model_to_bytes(model: CatBoostClassifier) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.save_model(tmp_path, format="cbm")
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return data


def upload_and_register(
    artifact_bytes: bytes,
    tr: TrainResult,
    make_current: bool = True
) -> None:
    """Upload the model file to Supabase Storage, then add it to model_registry.

    If the upload fails, the registry is left untouched.
    """
    sb = get_supabase()
    sb.storage.from_(MODEL_BUCKET).upload(model_object_name(tr.version), artifact_bytes)

    sb.table("model_registry").insert({
        "version": tr.version,
        "algo": "catboost",
        "params": {**tr.params, "feature_version": tr.feature_version, "cv_folds": tr.cv_folds},
        "metrics": tr.metrics,
        "samples_used": tr.n_samples,
        "is_current": make_current,
    }).execute()

    if make_current:
        # Retire the previous current model only after the new one is in place,
        # so the calculator never sees "no current model".
        (sb.table("model_registry")
           .update({"is_current": False})
           .eq("is_current", True)
           .neq("version", tr.version)
           .execute())
