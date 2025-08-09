# predict_service.py
import os, io
import numpy as np
import streamlit as st
from catboost import CatBoostClassifier
from db_supabase import get_supabase
from converting_service import convert  # keep your converter

FEATURE_COLUMNS = [
    "relapse","vegfa634gg","vegfa634c","periods","tp53gg","mecho",
    "vegfa936cc","first_symptom","kitlg80441cc","emergency_birth",
    "vleft_new","fsh","vright_new"
]

@st.cache_resource
def _load_current_model() -> CatBoostClassifier:
    sb = get_supabase()

    # 1) Find current model
    q = sb.table("model_registry").select("*").eq("is_current", True).limit(1).execute()
    items = q.data or []
    if not items:
        raise RuntimeError("No current model in registry. Train and set one as current.")

    version = items[0]["version"]
    # 2) Download artifact from Storage
    obj_name = f"{version}-catboost.cbm"
    # if alternate name was stored, you could store the exact object path in registry
    data = sb.storage.from_("models").download(obj_name)
    if hasattr(data, "content"):
        content = data.content  # supabase-py >= 2
    else:
        content = data  # supabase-py older versions

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
