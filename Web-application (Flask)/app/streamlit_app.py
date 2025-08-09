# streamlit_app.py  ─────────────────────────────────────────────────
import streamlit as st
import pandas as pd
from datetime import date
from converting_service import convert
from predict_service import predict

# NEW: import Supabase helpers
from db import insert_prediction, read_predictions

st.set_page_config("OMM Predict", page_icon="🩺", layout="centered")

# ────────────────────  0. Session state helpers  ──────────────────── #
if "records" not in st.session_state:
    st.session_state.records = []  # still keep local session history if you want

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

        # Keep session display if you like
        add_record(dict(
            id=len(st.session_state.records)+1,
            patient_card=patient_card,
            date_research=date.today().strftime("%d.%m.%Y"),
            target="Высокий" if outcome else "Низкий"
        ))

        # NEW: Persist to Supabase
        row = dict(
            patient_card   = patient_card,
            date_research  = date.today().strftime("%d.%m.%Y"),
            relapse        = int(relapse),
            periods        = float(periods),
            mecho          = float(mecho),
            first_symptom  = float(first_symptom),
            emergency_birth= int(emergency_birth),
            fsh            = float(fsh),
            vleft          = float(vleft),
            vright         = float(vright),
            vegfa634       = vegfa634,
            tp53           = tp53,
            vegfa936       = vegfa936,
            kitlg80441     = kitlg80441,
            outcome        = "Высокий" if outcome else "Низкий",
        )
        try:
            insert_prediction(row)
            st.success("**Высокий риск**" if outcome else "**Низкий риск**")
        except Exception as e:
            st.error(f"Ошибка записи в Supabase: {e}")

# =============  TAB 2 – session history (CSV export)  ============== #
with tab_history:
    st.header("История (из БД)")
    df_db = read_predictions(limit=1000)
    st.dataframe(df_db, use_container_width=True)

    st.subheader("Маркировка факта (истина/ложь)")
    # Work on a copy with only what we need to edit
    label_cols = ["id", "patient_card", "date_research", "outcome", "actual", "notes"]
    df_label = df_db[label_cols].copy() if not df_db.empty else pd.DataFrame(columns=label_cols)

    # Show only unlabeled first (you can toggle to show all)
    show_all = st.checkbox("Показать все записи", value=False)
    if not show_all:
        df_label = df_label[df_label["actual"].isna()]

    # Editable grid for 'actual' + 'notes'
    edited = st.data_editor(
        df_label,
        use_container_width=True,
        hide_index=True,
        column_config={
            "actual": st.column_config.SelectboxColumn(
                "Факт", options=["Высокий", "Низкий"], help="Подтвердите реальный исход"
            ),
            "notes": st.column_config.TextColumn("Заметки"),
            "outcome": st.column_config.TextColumn("Предсказание", disabled=True),
            "patient_card": st.column_config.TextColumn("Пациент", disabled=True),
            "date_research": st.column_config.TextColumn("Дата", disabled=True),
            "id": st.column_config.NumberColumn("ID", disabled=True),
        },
        disabled=["id", "patient_card", "date_research", "outcome"],
        key="label_editor",
    )

    # Save changes
    if st.button("Сохранить метки"):
        changed = 0
        # Compare row-by-row to original and push updates where actual/notes changed
        orig = df_db.set_index("id")
        for _, row in edited.iterrows():
            pid = int(row["id"])
            new_actual = row.get("actual")
            new_notes  = row.get("notes")
            old_actual = orig.at[pid, "actual"] if pid in orig.index else None
            old_notes  = orig.at[pid, "notes"]  if pid in orig.index else None

            if (pd.isna(old_actual) and pd.notna(new_actual)) or (old_actual != new_actual) or (str(old_notes) != str(new_notes)):
                try:
                    update_actual(pid, new_actual, new_notes)
                    changed += 1
                except Exception as e:
                    st.error(f"Ошибка обновления ID={pid}: {e}")

        if changed:
            st.success(f"Обновлено записей: {changed}")
            st.experimental_rerun()
        else:
            st.info("Нет изменений для сохранения.")
