# predict_service.py
import os, io, tempfile
import numpy as np
import streamlit as st
from catboost import CatBoostClassifier
from db import get_supabase
from converting_service import convert  # keep your converter

FEATURE_COLUMNS = [
    "relapse","vegfa634gg","vegfa634c","periods","tp53gg","mecho",
    "vegfa936cc","first_symptom","kitlg80441cc","emergency_birth",
    "vleft_new","fsh","vright_new"
]

class NoCurrentModel(Exception):
    """Raised when there is no current model set in the registry."""

@st.cache_data(ttl=15)
def _current_model_version() -> str | None:
    sb = get_supabase()
    q = sb.table("model_registry").select("version").eq("is_current", True).limit(1).execute()
    items = q.data or []
    return items[0]["version"] if items else None

def model_ready() -> bool:
    return _current_model_version() is not None

@st.cache_resource
def _load_current_model():
    sb = get_supabase()

    # 1) Find current model version
    q = sb.table("model_registry").select("*").eq("is_current", True).limit(1).execute()
    items = q.data or []
    if not items:
        raise RuntimeError("No current model in registry. Train and set one as current.")
    version = items[0]["version"]

    # 2) Download artifact bytes
    obj_name = f"{version}-catboost.cbm"
    data = sb.storage.from_("models").download(obj_name)

    # supabase-py returns raw bytes; some versions return dicts
    content = data.get("data") if isinstance(data, dict) else data
    if not isinstance(content, (bytes, bytearray)) or len(content) == 0:
        raise RuntimeError(f"Failed to download model bytes for {obj_name}")

    # 3) Write to a temp file and load
    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name

    try:
        model = CatBoostClassifier()
        model.load_model(tmp_path)  # must be a path
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return model

def _featurize_input(
    relapse, vegfa634, tp53, vegfa936, kitlg80441, periods, mecho,
    first_symptom, emergency_birth, fsh, vleft, vright
):
    vegfa634gg, vegfa634c, tp53gg, vegfa936cc, kitlg80441cc = convert(
        vegfa634, tp53, vegfa936, kitlg80441
    )
    vleft_new  = float(vleft)  * 532.0
    vright_new = float(vright) * 532.0

    row = {
        "relapse": float(relapse),
        "vegfa634gg": float(vegfa634gg),
        "vegfa634c": float(vegfa634c),
        "periods": float(periods),
        "tp53gg": float(tp53gg),
        "mecho": float(mecho),
        "vegfa936cc": float(vegfa936cc),
        "first_symptom": float(first_symptom),
        "kitlg80441cc": float(kitlg80441cc),
        "emergency_birth": float(emergency_birth),
        "vleft_new": float(vleft_new),
        "fsh": float(fsh),
        "vright_new": float(vright_new),
    }
    return np.array([[row[c] for c in FEATURE_COLUMNS]], dtype=float)

def predict(
    relapse, vegfa634, tp53, vegfa936, kitlg80441, periods, mecho,
    first_symptom, emergency_birth, fsh, vleft, vright,
    threshold: float = 0.5
) -> int:
    model = _load_current_model()
    X = _featurize_input(
        relapse, vegfa634, tp53, vegfa936, kitlg80441, periods, mecho,
        first_symptom, emergency_birth, fsh, vleft, vright
    )
    proba = model.predict_proba(X)[0, 1]
    return int(proba >= threshold)
