# streamlit_app.py  ─────────────────────────────────────────────────
import streamlit as st
import pandas as pd
from datetime import date
from converting_service import convert
from predict_service import predict
import streamlit as st
from db import get_conn, init_db, insert_prediction, read_predictions

@st.cache_resource
def get_db():
    conn = get_conn()
    init_db(conn)
    return conn

conn = get_db()

st.set_page_config("OMM Predict", page_icon="🩺", layout="centered")

# ────────────────────  0. Session state helpers  ──────────────────── #
if "records" not in st.session_state:
    st.session_state.records = []       # will hold dicts per patient

def add_record(rec: dict):
    st.session_state.records.append(rec)

# ────────────────────  1. Tabs  ───────────────────────────────────── #
tab_predict, tab_history = st.tabs(["🩺 Калькулятор", "📜 История сеанса"])

# =============  TAB 1 – single-patient calculator  ================= #
with tab_predict:
    st.title("OMM Predict – снижение овариального резерва")

    with st.form("calc"):
        c1, c2 = st.columns([2, 1])

        with c1:
            patient_card   = st.text_input("ФИО пациента")
            relapse        = st.selectbox("Рецидив эндометриомы", [0, 1])
            periods        = st.number_input("Менструация, дней", min_value=0.0, step=0.1)
            mecho          = st.number_input("М-эхо, мм", min_value=0.0, step=0.1)
            first_symptom  = st.number_input("Появление первых симптомов (лет)", min_value=0.0, step=0.1)
            emergency_birth= st.number_input("Срочные оперативные роды (кол-во)", min_value=0.0, step=1.0)
            fsh            = st.number_input("ФСГ до операции, мМе/мл", min_value=0.0, step=0.1)
            vleft          = st.number_input("V левого яичника, см³", min_value=0.0, step=0.1)
            vright         = st.number_input("V правого яичника, см³", min_value=0.0, step=0.1)

        with c2:
            vegfa634   = st.selectbox("VEGF-A −634", ["CC", "GC", "GG"])
            tp53       = st.selectbox("TP53 Ex4+119", ["CC", "CG", "GG"])
            vegfa936   = st.selectbox("VEGF-A +936", ["CC", "CT", "TT"])
            kitlg80441 = st.selectbox("KITLG 80441", ["CC", "CT", "TT"])

        submitted = st.form_submit_button("Рассчитать")

    if submitted:
        vegfa634gg, vegfa634c, tp53gg, vegfa936cc, kitlg80441cc = convert(
            vegfa634, tp53, vegfa936, kitlg80441
        )

        vleft_new  = vleft  * 532
        vright_new = vright * 532

        outcome = predict(
            relapse, vegfa634gg, vegfa634c, periods, tp53gg, mecho,
            vegfa936cc, first_symptom, kitlg80441cc, emergency_birth,
            vleft_new, fsh, vright_new
        )

        add_record( dict(
            id=len(st.session_state.records)+1,
            patient_card=patient_card,
            date_research=date.today().strftime("%d.%m.%Y"),
            target="Высокий" if outcome else "Низкий"
        ))

        st.success("**Высокий риск**" if outcome else "**Низкий риск**")
        row = dict(
            patient_card=patient_card,
            date_research=date.today().strftime("%d.%m.%Y"),
            relapse=int(relapse),
            periods=float(periods),
            mecho=float(mecho),
            first_symptom=float(first_symptom),
            emergency_birth=int(emergency_birth),
            fsh=float(fsh),
            vleft=float(vleft),
            vright=float(vright),
            vegfa634=vegfa634,
            tp53=tp53,
            vegfa936=vegfa936,
            kitlg80441=kitlg80441,
            outcome=("Высокий" if outcome else "Низкий"),
        )
        insert_prediction(conn, row)

# =============  TAB 2 – session history (CSV export)  ============== #
with tab_history:
    st.header("Теклая сессия")
    df = pd.DataFrame(st.session_state.records)
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Скачать CSV", csv, "ommpredict_history.csv", "text/csv")

# ─────────────────────────  END  ──────────────────────────────────── #
