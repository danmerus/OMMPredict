# streamlit_app.py  ─────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from converting_service import convert
from predict_service import predict, model_ready, NoCurrentModel

# NEW: import Supabase helpers
from db import insert_prediction, read_predictions, update_actual

st.set_page_config("OMM Predict", page_icon="🩺", layout="centered")

# ────────────────────  0. Session state helpers  ──────────────────── #
if "records" not in st.session_state:
    st.session_state.records = []  # still keep local session history if you want

def add_record(rec: dict):
    st.session_state.records.append(rec)

# ────────────────────  1. Tabs  ───────────────────────────────────── #
tab_predict, tab_history, tab_train = st.tabs(["🩺 Калькулятор", "📜 История сеанса", "🧠 Обучение"])

# =============  TAB 1 – single-patient calculator  ================= #
with tab_predict:
    st.title("OMM Predict – снижение овариального резерва")

    with st.form("calc"):
        c1, c2 = st.columns([2, 1])

        with c1:
            patient_card    = st.text_input("ФИО пациента")
            relapse         = st.selectbox("Рецидив эндометриомы", [0, 1])
            periods         = st.number_input("Менструация, дней", min_value=0.0, step=0.1)
            mecho           = st.number_input("М-эхо, мм", min_value=0.0, step=0.1)
            first_symptom   = st.number_input("Появление первых симптомов (лет)", min_value=0.0, step=0.1)
            emergency_birth = st.number_input("Срочные оперативные роды (кол-во)", min_value=0.0, step=1.0)
            fsh             = st.number_input("ФСГ до операции, мМе/мл", min_value=0.0, step=0.1)
            vleft           = st.number_input("V левого яичника, см³", min_value=0.0, step=0.1)
            vright          = st.number_input("V правого яичника, см³", min_value=0.0, step=0.1)

        with c2:
            vegfa634   = st.selectbox("VEGF-A −634", ["CC", "GC", "GG"])
            tp53       = st.selectbox("TP53 Ex4+119", ["CC", "CG", "GG"])
            vegfa936   = st.selectbox("VEGF-A +936", ["CC", "CT", "TT"])
            kitlg80441 = st.selectbox("KITLG 80441", ["CC", "CT", "TT"])

        submitted = st.form_submit_button("Рассчитать")
    SCALE = 532.0 # for volume
    if submitted:
        try:
            if not model_ready():
                st.info("Модель пока не выбрана. Перейдите во вкладку **🧠 Обучение**, обучите модель и отметьте её как текущую.")
            else:
                outcome = predict(
                    int(relapse), vegfa634, tp53, vegfa936, kitlg80441,
                    float(periods), float(mecho), float(first_symptom), int(emergency_birth),
                    float(fsh), float(vleft)*SCALE, float(vright)*SCALE
                )
                risk_text = "**Высокий риск**" if outcome else "**Низкий риск**"
                st.success(risk_text)

                # Save to session
                add_record(dict(
                    id=len(st.session_state.records)+1,
                    patient_card=patient_card,
                    date_research=date.today().strftime("%d.%m.%Y"),
                    target="Высокий" if outcome else "Низкий"
                ))

                # Persist to Supabase
                row = dict(
                    patient_card    = patient_card,
                    date_research   = date.today().strftime("%d.%m.%Y"),
                    relapse         = int(relapse),
                    periods         = float(periods),
                    mecho           = float(mecho),
                    first_symptom   = float(first_symptom),
                    emergency_birth = int(emergency_birth),
                    fsh             = float(fsh),
                    vleft           = float(vleft),
                    vright          = float(vright),
                    vegfa634        = vegfa634,
                    tp53            = tp53,
                    vegfa936        = vegfa936,
                    kitlg80441      = kitlg80441,
                    outcome         = "Высокий" if outcome else "Низкий",
                )
                try:
                    insert_prediction(row)
                    st.caption("Запись сохранена в Supabase.")
                except Exception as e:
                    st.warning(f"Не удалось сохранить в Supabase: {e}")

        except NoCurrentModel as e:
            st.info("Модель ещё не настроена. Откройте вкладку **🧠 Обучение** и создайте модель.")
            with st.expander("Подробности (для разработчика)"):
                st.write(str(e))
        except Exception as e:
            st.error("Не удалось сделать прогноз. Попробуйте позже или переобучите модель.")
            with st.expander("Подробности (для разработчика)"):
                st.exception(e)

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
    
with tab_train:
    st.title("🧠 Переобучение модели")

    st.markdown("Выберите источники данных и параметры обучения.")

    c1, c2 = st.columns(2)
    # with c1:
    #     include_local = st.checkbox("Добавить локальные baseline-примеры", value=False,
    #                                 help="CSV/XLSX с такими полями, как в БД (или минимум: признаки + actual).")
    #     local_path = st.text_input("Путь к baseline файлу", value="data/baseline_examples.csv")
    include_local = True
    local_path = ''

    with c2:
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random seed", 1, 999999, 42)

    st.subheader("CatBoost параметры")
    c3, c4, c5, c6 = st.columns(4)
    with c3:
        iterations = st.number_input("iterations", 100, 5000, 600, 50)
    with c4:
        learning_rate = st.number_input("learning_rate", 0.001, 0.5, 0.05, 0.001)
    with c5:
        depth = st.number_input("depth", 3, 10, 6, 1)
    with c6:
        l2_leaf_reg = st.number_input("l2_leaf_reg", 1.0, 20.0, 3.0, 0.5)

    st.divider()
    from train_service import (
        assemble_training_data, train_catboost,
        save_model_to_bytes, upload_and_register
    )

    if st.button("🚀 Запустить обучение"):
        with st.spinner("Готовим данные..."):
            df = assemble_training_data(include_local, local_path)

        if len(df) < 30 or "target" not in df.columns:
            st.error("Недостаточно размеченных данных. Отметьте фактические исходы во вкладке История.")
        else:
            st.success(f"Найдено обучающих примеров: {len(df)}")
            st.write("Примеры признаков:", df.head(3))

            with st.spinner("Обучаем CatBoost..."):
                tr = train_catboost(
                    df,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    iterations=int(iterations),
                    learning_rate=float(learning_rate),
                    depth=int(depth),
                    l2_leaf_reg=float(l2_leaf_reg),
                )

            st.subheader("Метрики")
            m = tr.metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{m['accuracy']:.3f}")
            c2.metric("Precision", f"{m['precision']:.3f}")
            c3.metric("Recall", f"{m['recall']:.3f}")
            c4.metric("F1", f"{m['f1']:.3f}")
            c5.metric("AUC (val)", f"{m['auc']:.3f}")

            # Confusion matrix plot
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.imshow(tr.confusion, interpolation="nearest")
            plt.title("Confusion matrix")
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ["0", "1"])
            plt.yticks(tick_marks, ["0", "1"])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, tr.confusion[i, j], ha="center", va="center")
            st.pyplot(fig)

            # Download artifact
            artifact = save_model_to_bytes(tr.model)
            st.download_button(
                "💾 Скачать модель (.cbm)",
                data=artifact,
                file_name=f"{tr.version}-catboost.cbm",
                mime="application/octet-stream"
            )

            st.subheader("Сохранить модель как текущую")
            make_current = st.checkbox("Сделать текущей (будет использоваться в калькуляторе)", value=True)
            if st.button("📤 Загрузить в Supabase и зарегистрировать"):
                try:
                    with st.spinner("Загрузка артефакта и запись в реестр..."):
                        upload_and_register(artifact, tr, make_current=make_current)
                    st.success("Модель сохранена и зарегистрирована!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка при загрузке/регистрации: {e}")

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
            st.rerun()
        else:
            st.info("Нет изменений для сохранения.")
