from supabase import create_client, Client
import streamlit as st
import pandas as pd
from datetime import datetime, timezone

@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["supabase"]["url"]
    # key = st.secrets["supabase"].get("service_key") or st.secrets["supabase"]["anon_key"]
    key = st.secrets["supabase"]["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)

def insert_prediction(row: dict):
    sb = get_supabase()
    return sb.table("predictions").insert(row).execute()

def read_predictions(limit: int = 1000) -> pd.DataFrame:
    sb = get_supabase()
    res = (sb.table("predictions")
             .select("*")
             .order("id", desc=True)
             .limit(limit)
             .execute())
    return pd.DataFrame(res.data or [])

def update_actual(id_: int, actual: str | None, notes: str | None):
    sb = get_supabase()
    payload = {
        "actual": actual,
        "notes": notes,
        "verified_at": datetime.now(timezone.utc).isoformat()
    }
    return (sb.table("predictions")
              .update(payload)
              .eq("id", id_)
              .execute())
