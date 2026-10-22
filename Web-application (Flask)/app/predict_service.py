# predict_service.py
import json
import os
import tempfile

import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

from db import MODEL_BUCKET, get_supabase, model_object_name
from features import FEATURE_VERSION, build_features


class NoCurrentModel(Exception):
    """Raised when there is no current model set in the registry."""


class OutdatedModel(Exception):
    """Raised when the current model was trained with a different feature format."""


@st.cache_data(ttl=15)
def current_model_info() -> dict | None:
    """Registry row ({version, params}) of the current model, or None."""
    sb = get_supabase()
    q = sb.table("model_registry").select("version, params").eq("is_current", True).limit(1).execute()
    items = q.data or []
    return items[0] if items else None


def model_ready() -> bool:
    return current_model_info() is not None


@st.cache_resource(max_entries=3)
def _load_model(version: str) -> CatBoostClassifier:
    """Download and load one model version. Cached per version, so a newly
    registered model is picked up as soon as the registry points at it."""
    obj_name = model_object_name(version)
    data = get_supabase().storage.from_(MODEL_BUCKET).download(obj_name)

    # supabase-py returns raw bytes; some versions return dicts
    content = data.get("data") if isinstance(data, dict) else data
    if not isinstance(content, (bytes, bytearray)) or len(content) == 0:
        raise RuntimeError(f"Failed to download model bytes for {obj_name}")

    # CatBoost loads from a path, so go through a temp file
    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        model = CatBoostClassifier()
        model.load_model(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return model


def predict(patient: dict, threshold: float = 0.5) -> tuple[int, float]:
    """Predict for one patient given in canonical units (volumes in cm³,
    genotypes as letters). Returns (1 = high risk / 0 = low, probability)."""
    info = current_model_info()
    if info is None:
        raise NoCurrentModel("No current model in registry. Train and set one as current.")

    params = info.get("params") or {}
    if isinstance(params, str):
        params = json.loads(params)
    model_fv = params.get("feature_version")
    if model_fv != FEATURE_VERSION:
        raise OutdatedModel(
            f"Model {info['version']} was trained with feature_version={model_fv}; "
            f"the app now uses feature_version={FEATURE_VERSION}."
        )

    X = build_features(pd.DataFrame([patient]))
    missing = X.columns[X.isna().any()].tolist()
    if missing:
        raise ValueError(f"Invalid or missing inputs: {missing}")

    model = _load_model(info["version"])
    proba = float(model.predict_proba(X.values)[0, 1])
    return int(proba >= threshold), proba
