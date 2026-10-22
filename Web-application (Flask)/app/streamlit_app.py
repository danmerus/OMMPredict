# streamlit_app.py  ─────────────────────────────────────────────────
import hmac
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from db import get_supabase, insert_prediction, read_predictions, update_actual
from labeling import changed_labels
from model_registry import describe, is_compatible, models_table, set_current_model
from predict_service import (
    NoCurrentModel, OutdatedModel, current_model_info, list_models, model_ready, predict,
    refresh_registry_cache,
)
from train_service import (
    assemble_training_data, save_model_to_bytes, train_catboost, upload_and_register,
)

st.set_page_config("OMM Predict", page_icon="🩺", layout="centered")


# ────────────────────  0. Access control  ─────────────────────────── #
def require_password() -> None:
    """Shared-password gate. Fails closed when no password is configured."""
    if st.session_state.get("authenticated"):
        return
    try:
        expected = st.secrets["auth"]["password"]
    except Exception:
        expected = None
    if not expected:
        st.error("Доступ закрыт: пароль приложения не настроен (Streamlit secrets → `[auth] password`).")
        st.stop()

    st.title("OMM Predict")
    with st.form("login"):
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Войти")
    if submitted:
        # compare bytes: compare_digest rejects non-ASCII str
        if hmac.compare_digest(password.encode("utf-8"), str(expected).encode("utf-8")):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Неверный пароль")
    st.stop()


require_password()

# ────────────────────  1. Session state helpers  ──────────────────── #
if "records" not in st.session_state:
    st.session_state.records = []
if "tr" not in st.session_state: st.session_state["tr"] = None
if "artifact" not in st.session_state: st.session_state["artifact"] = None

def add_record(rec: dict):
    st.session_state.records.append(rec)

# ────────────────────  2. Tabs  ───────────────────────────────────── #
tab_predict, tab_history, tab_train = st.tabs(["🩺 Калькулятор", "📜 История сеанса", "🧠 Обучение"])

# =============  TAB 1 – single-patient calculator  ================= #
with tab_predict:
    st.title("OMM Predict – снижение овариального резерва")
    try:
        _info = current_model_info()
        st.caption(f"Текущая модель: {describe(_info)}" if _info else "Текущая модель: не выбрана")
    except Exception:
        st.caption("Текущая модель: не удалось получить данные из реестра")

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
            tp53       = st.selectbox("TP53 Ex4+119", ["CC", "GC", "GG"])
            vegfa936   = st.selectbox("VEGF-A +936", ["CC", "CT", "TT"])
            kitlg80441 = st.selectbox("KITLG 80441", ["CC", "CT", "TT"])

        submitted = st.form_submit_button("Рассчитать")

    if submitted:
        # Canonical units (cm³, genotype letters): the model gets exactly what is stored in the DB.
        inputs = dict(
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
        )
        try:
            if not model_ready():
                st.info("Модель пока не выбрана. Перейдите во вкладку **🧠 Обучение**, обучите модель и отметьте её как текущую.")
            else:
                outcome, proba = predict(inputs)
                risk_text = "**Высокий риск**" if outcome else "**Низкий риск**"
                st.success(risk_text)
                st.caption(f"Оценка модели: {proba:.0%} (порог «высокого риска» — 50%; это не калиброванная вероятность).")

                # Save to session
                add_record(dict(
                    id=len(st.session_state.records)+1,
                    patient_card=patient_card,
                    date_research=date.today().strftime("%d.%m.%Y"),
                    target="Высокий" if outcome else "Низкий"
                ))

                # Persist to Supabase
                row = dict(
                    patient_card  = patient_card,
                    date_research = date.today().strftime("%d.%m.%Y"),
                    **inputs,
                    outcome       = "Высокий" if outcome else "Низкий",
                )
                try:
                    insert_prediction(row)
                    st.caption("Запись сохранена в Supabase.")
                except Exception as e:
                    st.warning(f"Не удалось сохранить в Supabase: {e}")

        except OutdatedModel as e:
            st.warning("Текущая модель обучена в старом формате данных и больше не подходит. "
                       "Откройте вкладку **🧠 Обучение**, обучите модель и сделайте её текущей.")
            with st.expander("Подробности (для разработчика)"):
                st.write(str(e))
        except NoCurrentModel as e:
            st.info("Модель ещё не настроена. Откройте вкладку **🧠 Обучение** и создайте модель.")
            with st.expander("Подробности (для разработчика)"):
                st.write(str(e))
        except Exception as e:
            st.error("Не удалось сделать прогноз. Попробуйте позже или переобучите модель.")
            with st.expander("Подробности (для разработчика)"):
                st.exception(e)

# =============  TAB 2 – history + outcome labelling  =============== #
with tab_history:
    st.header("История (из БД)")
    df_db = read_predictions(limit=1000)
    st.dataframe(df_db, use_container_width=True)

    st.subheader("Маркировка факта (истина/ложь)")
    saved = st.session_state.pop("labels_saved", None)
    if saved:
        st.success(f"Сохранено записей: {saved}")

    label_cols = ["id", "patient_card", "date_research", "outcome", "actual", "notes"]
    df_label = df_db[label_cols].copy() if not df_db.empty else pd.DataFrame(columns=label_cols)
    # All-empty columns can come back as float NaN, which the text/select editors refuse.
    df_label = df_label.astype({"actual": "object", "notes": "object"})

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
        key=f"label_editor_{show_all}",  # separate state per view, so edits never shift onto other rows
    )

    if st.button("💾 Сохранить метки"):
        changes = changed_labels(df_label, edited)
        if not changes:
            st.info("Нет изменений для сохранения.")
        else:
            errors = []
            for pid, actual, notes in changes:
                try:
                    update_actual(pid, actual, notes)
                except Exception as e:
                    errors.append(f"ID={pid}: {e}")
            # Edits are stored by row position; drop them so they can't land on other rows after reload.
            for view in (True, False):
                st.session_state.pop(f"label_editor_{view}", None)
            if errors:
                st.error("Ошибки при сохранении:\n\n" + "\n\n".join(errors))
            else:
                st.session_state["labels_saved"] = len(changes)
                st.rerun()

# =============  TAB 3 – training  ================================== #
with tab_train:
    st.title("🧠 Переобучение модели")

    st.markdown("Обучение на размеченных записях из БД (вкладка История) и базовых примерах.")

    include_local = True
    local_path = ''

    c2 = st.columns(1)[0]
    with c2:
        n_splits = st.slider("Фолды кросс-валидации", 3, 10, 5, 1)
        random_state = st.number_input("Random seed", 1, 999999, 42)

    st.subheader("CatBoost параметры")
    st.caption("Данных мало (~70 примеров), поэтому по умолчанию — небольшая модель: на кросс-валидации она точнее и не даёт крайних оценок.")
    c3, c4, c5, c6 = st.columns(4)
    with c3:
        iterations = st.number_input("iterations", 100, 5000, 150, 50)
    with c4:
        learning_rate = st.number_input("learning_rate", 0.001, 0.5, 0.03, 0.001)
    with c5:
        depth = st.number_input("depth", 3, 10, 3, 1)
    with c6:
        l2_leaf_reg = st.number_input("l2_leaf_reg", 1.0, 20.0, 3.0, 0.5)

    st.divider()
    flash = st.session_state.pop("model_registered", None)
    if flash:
        st.success(flash)

    if st.button("🚀 Запустить обучение"):
        with st.spinner("Готовим данные..."):
            df = assemble_training_data(include_local, local_path)
        counts = df.attrs.get("counts", {})
        st.caption(
            f"Размеченных записей из БД: {counts.get('db', 0)} · baseline: {counts.get('baseline', 0)} · "
            f"отброшено неполных: {counts.get('dropped_incomplete', 0)} · дублей: {counts.get('dropped_duplicates', 0)}"
        )

        if len(df) < 30 or "target" not in df.columns:
            st.error("Недостаточно размеченных данных. Отметьте фактические исходы во вкладке История.")
        else:
            st.success(f"Найдено обучающих примеров: {len(df)}")
            st.write("Примеры признаков:", df.head(3))

            try:
                with st.spinner("Обучаем CatBoost (кросс-валидация + итоговая модель)..."):
                    tr = train_catboost(
                        df,
                        n_splits=int(n_splits),
                        random_state=int(random_state),
                        iterations=int(iterations),
                        learning_rate=float(learning_rate),
                        depth=int(depth),
                        l2_leaf_reg=float(l2_leaf_reg),
                    )
            except ValueError as e:
                st.error(str(e))
            else:
                # store for later runs
                st.session_state.tr = tr
                st.session_state.artifact = save_model_to_bytes(tr.model)

    # --- Show results (if we have them), independent of the Train button state ---
    tr = st.session_state.tr
    artifact = st.session_state.artifact
    if tr is not None:
        st.subheader("Метрики (кросс-валидация)")
        st.caption(
            f"{tr.cv_folds}-кратная кросс-валидация по {tr.n_samples} примерам "
            f"(высокий риск: {tr.metrics['pos']}, низкий: {tr.metrics['neg']}). "
            "Итоговая модель обучена на всех примерах."
        )
        m = tr.metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['accuracy']:.3f}")
        c2.metric("Precision", f"{m['precision']:.3f}")
        c3.metric("Recall", f"{m['recall']:.3f}")
        c4.metric("F1", f"{m['f1']:.3f}")
        c5.metric("AUC (CV)", f"{m['auc']:.3f}")

        fig = plt.figure()
        plt.imshow(tr.confusion, interpolation="nearest"); plt.title("Confusion matrix (CV)"); plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["0", "1"]); plt.yticks(tick_marks, ["0", "1"])
        plt.xlabel("Predicted"); plt.ylabel("True")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, tr.confusion[i, j], ha="center", va="center")
        st.pyplot(fig)
        plt.close(fig)

        st.download_button("💾 Скачать модель (.cbm)",
            data=artifact, file_name=f"{tr.version}-catboost.cbm",
            mime="application/octet-stream")

        st.subheader("Сохранить модель как текущую")
        make_current = st.checkbox("Сделать текущей (будет использоваться в калькуляторе)", value=True, key="mkcur")

        if st.button("📤 Загрузить в Supabase и зарегистрировать", key="upload"):
            try:
                with st.spinner("Загрузка артефакта и запись в реестр..."):
                    upload_and_register(artifact, tr, make_current=make_current)
                refresh_registry_cache()  # calculator picks up the new model right away
                st.session_state["model_registered"] = f"Модель {tr.version} сохранена и зарегистрирована!"
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка при загрузке/регистрации: {e}")

    # --- Registered models: see them all and choose which one the calculator uses ---
    st.divider()
    st.subheader("📚 Зарегистрированные модели")
    switched = st.session_state.pop("model_switched", None)
    if switched:
        st.success(switched)

    try:
        models = list_models()
    except Exception as e:
        models = []
        st.error(f"Не удалось загрузить реестр моделей: {e}")

    if not models:
        st.info("Пока нет зарегистрированных моделей.")
    else:
        st.dataframe(
            models_table(models), use_container_width=True, hide_index=True,
            column_config={"AUC (CV)": st.column_config.NumberColumn(format="%.3f")},
        )
        st.caption(
            "AUC модели в старом формате не сравним и не показан. Разница AUC между переобучениями "
            "на ~70 примерах — в основном шум: выбирайте самую новую модель или обученную на большем "
            "числе примеров, а не ту, у которой AUC выше."
        )
        compatible = {m["version"]: m for m in models if is_compatible(m)}
        current = next((m["version"] for m in models if m.get("is_current")), None)
        if not compatible:
            st.info("Нет моделей в актуальном формате — обучите и зарегистрируйте модель выше.")
        else:
            versions = list(compatible)
            chosen = st.selectbox(
                "Модель для калькулятора",
                versions,
                index=versions.index(current) if current in compatible else 0,
                format_func=lambda v: describe(compatible[v]) + (" (текущая)" if v == current else ""),
                key="model_choice",
            )
            if st.button("✅ Сделать текущей", disabled=(chosen == current)):
                try:
                    set_current_model(get_supabase(), chosen)
                    refresh_registry_cache()
                    st.session_state["model_switched"] = f"Калькулятор теперь использует модель {chosen}."
                    st.rerun()
                except Exception as e:
                    st.error(f"Не удалось сменить модель: {e}")
