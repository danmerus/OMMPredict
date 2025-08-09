# train_service.py
from __future__ import annotations
import io, json, time, uuid, os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

import streamlit as st
from db import get_supabase  # from your earlier helper
from converting_service import convert  # your existing converter

# -------- Feature set used in your inference code --------
FEATURE_COLUMNS = [
    "relapse", "vegfa634gg", "vegfa634c", "periods", "tp53gg", "mecho",
    "vegfa936cc", "first_symptom", "kitlg80441cc", "emergency_birth",
    "vleft_new", "fsh", "vright_new"
]

GENO_COLUMNS = ["vegfa634", "tp53", "vegfa936", "kitlg80441"]

APP_DIR = Path(__file__).resolve().parent         # .../Web-application (Flask)/app
DATA_DIR = APP_DIR / "data"                       # .../app/data
DEFAULT_BASELINE = DATA_DIR / "baseline_examples.csv"

@dataclass
class TrainResult:
    model: CatBoostClassifier
    metrics: Dict[str, float]
    confusion: np.ndarray
    version: str
    params: Dict[str, Any]
    X_cols: List[str]
    n_samples: int

def _from_supabase_verified(limit: int = 50000) -> pd.DataFrame:
    """Load verified rows from Supabase predictions table."""
    sb = get_supabase()
    # Pull only rows where 'actual' is present and valid
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

    # Map Russian labels -> binary target
    # 'Высокий' -> 1, 'Низкий' -> 0
    lab_map = {"Высокий": 1, "Низкий": 0}
    df["target"] = df["actual"].map(lab_map)
    return df

def _ensure_target_column(df: pd.DataFrame) -> pd.DataFrame:
    LAB_MAP = {"Высокий": 1, "Низкий": 0, "high": 1, "low": 0, "1": 1, "0": 0}
    """Create/normalize numeric 'target' in df if possible."""
    if "target" in df.columns:
        # map strings to {0,1} if needed
        if df["target"].dtype == object:
            df["target"] = pd.to_numeric(df["target"].map(LAB_MAP), errors="coerce")
        else:
            df["target"] = pd.to_numeric(df["target"], errors="coerce")
        return df

    # try typical alternatives
    for cand in ("actual", "outcome", "label", "y", "class"):
        if cand in df.columns:
            col = df[cand]
            if col.dtype == object:
                df["target"] = pd.to_numeric(col.map(LAB_MAP), errors="coerce")
            else:
                df["target"] = pd.to_numeric(col, errors="coerce")
            return df

    return df


def _from_local_baseline(rel_or_abs: str | None = None) -> pd.DataFrame:
    if rel_or_abs:
        p = Path(rel_or_abs)
        p = p if p.is_absolute() else (APP_DIR / rel_or_abs)
    else:
        p = DEFAULT_BASELINE

    p = p.resolve()

    # Let user see where we looked
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

    n_raw = len(df)
    if n_raw == 0:
        st.warning(f"Файл пуст: {p}")
        return df

    # derive numeric target if possible
    df = _ensure_target_column(df)

    st.caption(f"✅ Загружено baseline строк: {n_raw} (столбцы: {list(df.columns)[:10]}...)")
    return df

def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same transforms used at inference time:
    - genotype conversion via convert()
    - volume * 532
    - ensure dtypes and columns exist
    """
    # Guard: columns that must exist (create if absent)
    for col in ["relapse","periods","mecho","first_symptom","emergency_birth","fsh","vleft","vright"] + GENO_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Convert categorical genotypes to binary features using your converter
    # convert(vegfa634, tp53, vegfa936, kitlg80441) -> vegfa634gg, vegfa634c, tp53gg, vegfa936cc, kitlg80441cc
    conv = df[GENO_COLUMNS].astype(str).apply(
        lambda r: convert(r["vegfa634"], r["tp53"], r["vegfa936"], r["kitlg80441"]),
        axis=1, result_type="expand"
    )
    conv.columns = ["vegfa634gg", "vegfa634c", "tp53gg", "vegfa936cc", "kitlg80441cc"]

    df = pd.concat([df, conv], axis=1)

    # Numerical transforms
    df["vleft_new"]  = pd.to_numeric(df["vleft"], errors="coerce")  * 532.0
    df["vright_new"] = pd.to_numeric(df["vright"], errors="coerce") * 532.0

    # Cast core numerics
    for c in ["relapse","periods","mecho","first_symptom","emergency_birth","fsh"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only required columns + target if present
    keep = FEATURE_COLUMNS + (["target"] if "target" in df.columns else [])
    return df[keep].copy()

def assemble_training_data(
    include_local: bool,
    local_path: Optional[str]
) -> pd.DataFrame:
    df_sb = _from_supabase_verified()
    df_sb = _apply_feature_engineering(df_sb)
    if include_local:
        df_base = _from_local_baseline(local_path or "")
        if not df_base.empty:
            df_base = _apply_feature_engineering(df_base)
            df = pd.concat([df_sb, df_base], ignore_index=True)
        else:
            df = df_sb
    else:
        df = df_sb

    # Drop rows with missing features/labels
    df = df.dropna(subset=FEATURE_COLUMNS + (["target"] if "target" in df.columns else []))
    df = df.drop_duplicates()
    return df

def train_catboost(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    iterations: int = 600,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0
) -> TrainResult:
    X = df[FEATURE_COLUMNS].astype(float)
    y = df["target"].astype(int)

    # Handle class imbalance with scale_pos_weight
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    spw = (neg / max(pos, 1)) if pos > 0 else 1.0

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique()==2 else None
    )

    train_pool = Pool(X_tr, y_tr)
    test_pool  = Pool(X_te, y_te)

    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_state,
        verbose=False,
        scale_pos_weight=float(spw),
        od_type="Iter", 
        od_wait=50 
    )

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool, verbose=False)

    # Predict using 0.5 threshold
    proba = model.predict_proba(X_te)[:, 1]
    y_hat = (proba >= 0.5).astype(int)

    metrics = dict(
        accuracy=float(accuracy_score(y_te, y_hat)),
        precision=float(precision_score(y_te, y_hat, zero_division=0)),
        recall=float(recall_score(y_te, y_hat, zero_division=0)),
        f1=float(f1_score(y_te, y_hat, zero_division=0)),
        auc=float(model.get_best_score().get("validation", {}).get("AUC", np.nan)),
        pos= int(pos),
        neg= int(neg)
    )
    cm = confusion_matrix(y_te, y_hat)

    version = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    return TrainResult(
        model=model,
        metrics=metrics,
        confusion=cm,
        version=version,
        params=params,
        X_cols=FEATURE_COLUMNS,
        n_samples=len(df)
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
    """Upload model to Supabase Storage and insert a row into model_registry."""
    sb = get_supabase()
    bucket = "models"
    obj_name = f"{tr.version}-catboost.cbm"

    # Upload (overwrite=False)
    # If object exists, add a random suffix
    try:
        sb.storage.from_(bucket).upload(obj_name, artifact_bytes)
    except Exception:
        obj_name = f"{tr.version}-{uuid.uuid4().hex[:8]}-catboost.cbm"
        sb.storage.from_(bucket).upload(obj_name, artifact_bytes)

    # If make_current=True, unset previous current
    if make_current:
        sb.table("model_registry").update({"is_current": False}).eq("is_current", True).execute()

    sb.table("model_registry").insert({
        "version": tr.version,
        "algo": "catboost",
        "params": tr.params,
        "metrics": tr.metrics,
        "samples_used": tr.n_samples,
        "is_current": make_current
    }).execute()
