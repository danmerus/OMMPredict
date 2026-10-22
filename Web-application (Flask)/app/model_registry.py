"""Helpers for rows of the `model_registry` table (no Streamlit / Supabase imports)."""
from __future__ import annotations

import json

import pandas as pd

from features import FEATURE_VERSION


def _as_dict(value) -> dict:
    """`params` / `metrics` may arrive as a dict, a JSON string or None."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def params_of(row: dict) -> dict:
    return _as_dict(row.get("params"))


def metrics_of(row: dict) -> dict:
    return _as_dict(row.get("metrics"))


def feature_version_of(row: dict):
    return params_of(row).get("feature_version")


def is_compatible(row: dict) -> bool:
    """True if the model was trained with the feature format this app uses."""
    return feature_version_of(row) == FEATURE_VERSION


def cv_auc(row: dict) -> float | None:
    """Cross-validated AUC. Old-format models stored a different (validation) AUC, so they get None."""
    if not is_compatible(row):
        return None
    auc = metrics_of(row).get("auc")
    try:
        return float(auc)
    except (TypeError, ValueError):
        return None


def describe(row: dict) -> str:
    """One-line summary, e.g. '2026-10-22T07-48-32Z-9505b4 · AUC (CV) 0.75 · 69 примеров'."""
    parts = [str(row.get("version"))]
    auc = cv_auc(row)
    if auc is not None:
        parts.append(f"AUC (CV) {auc:.2f}")
    if row.get("samples_used") is not None:
        parts.append(f"{row['samples_used']} примеров")
    if not is_compatible(row):
        parts.append("старый формат")
    return " · ".join(parts)


def set_current_model(sb, version: str) -> None:
    """Make `version` the model the calculator uses.

    Checks the version first: an unknown or old-format version raises and
    changes nothing, so a mistake can never leave the calculator without a model.
    The new model is switched on before the old one is switched off.
    """
    rows = (sb.table("model_registry").select("version, params")
              .eq("version", version).limit(1).execute().data) or []
    if not rows:
        raise ValueError(f"Модель {version} не найдена в реестре.")
    if not is_compatible(rows[0]):
        raise ValueError(f"Модель {version} обучена в старом формате данных и не может использоваться.")
    sb.table("model_registry").update({"is_current": True}).eq("version", version).execute()
    (sb.table("model_registry").update({"is_current": False})
       .eq("is_current", True).neq("version", version).execute())


def models_table(rows: list[dict]) -> pd.DataFrame:
    """Registry rows -> table for display, newest first."""
    records = []
    for r in sorted(rows, key=lambda r: str(r.get("version")), reverse=True):
        p = params_of(r)
        records.append({
            "Версия": r.get("version"),
            "Текущая": bool(r.get("is_current")),
            "AUC (CV)": cv_auc(r),
            "Примеров": r.get("samples_used"),
            "depth": p.get("depth"),
            "iterations": p.get("iterations"),
            "learning_rate": p.get("learning_rate"),
            "Совместима": is_compatible(r),
        })
    columns = ["Версия", "Текущая", "AUC (CV)", "Примеров", "depth", "iterations", "learning_rate", "Совместима"]
    return pd.DataFrame(records, columns=columns)
