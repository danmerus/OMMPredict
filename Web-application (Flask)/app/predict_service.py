# predict_service.py
import os, io
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
def _load_current_model() -> CatBoostClassifier:
    sb = get_supabase()
    version = _current_model_version()
    if not version:
        # <- raise custom, not generic RuntimeError
        raise NoCurrentModel("No current model is set. Please train one in the 'Обучение' tab.")
    obj_name = f"{version}-catboost.cbm"
    data = sb.storage.from_("models").download(obj_name)
    content = getattr(data, "content", data)
    model = CatBoostClassifier()
    model.load_model(io.BytesIO(content))
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
