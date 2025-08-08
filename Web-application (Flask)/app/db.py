from supabase import create_client, Client
import streamlit as st
import pandas as pd

@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["supabase"]["url"]
    # Use service_key if available; else fallback to anon_key
    key = st.secrets["supabase"].get("service_key") or st.secrets["supabase"]["anon_key"]
    return create_client(url, key)

def insert_prediction(row: dict):
    sb = get_supabase()
    # supabase-py expects JSON-serializable primitives
    res = sb.table("predictions").insert(row).execute()
    # res.data contains inserted rows (list of dicts)
    return res

def read_predictions(limit: int = 1000) -> pd.DataFrame:
    sb = get_supabase()
    res = sb.table("predictions") \
            .select("*") \
            .order("id", desc=True) \
            .limit(limit) \
            .execute()
    return pd.DataFrame(res.data or [])
