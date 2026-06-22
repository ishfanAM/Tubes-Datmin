import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Produktivitas Digital", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("xgboost_model.pkl")
    kolom_model = joblib.load("model_columns.pkl")
    return model, kolom_model

model, kolom_model = load_model()

st.title("Prediksi Hubungan Penggunaan Sosial Media dengan Produktivitas")
st.caption("Prototipe Deployment - TUBES Penambangan Data | XGBoost Champion Model")

col_input, col_hasil = st.columns([1, 1])

with col_input:
    st.subheader("Panel Input")
    age = st.slider("Usia", 18, 65, 25)
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male", "Other"])
    job_type = st.selectbox("Tipe Pekerjaan", ["Education", "Finance", "Health", "IT", "Student", "Unemployed"])
    platform = st.selectbox("Platform Sosial Media Favorit", ["Facebook", "Instagram", "Telegram", "TikTok", "Twitter"])
    daily_social_media_time = st.slider("Waktu Sosial Media Harian (jam)", 0.0, 12.0, 3.5, 0.1)
    number_of_notifications = st.slider("Jumlah Notifikasi per Hari", 30, 90, 55)
    work_hours_per_day = st.slider("Jam Kerja per Hari", 0.0, 12.0, 8.0, 0.5)
    perceived_productivity_score = st.slider("Skor Produktivitas yang Dirasakan (1-10)", 1.0, 10.0, 6.0, 0.1)
    stress_level = st.slider("Tingkat Stres (1-10)", 1.0, 10.0, 5.0, 0.5)
    sleep_hours = st.slider("Jam Tidur per Hari", 3.0, 10.0, 7.0, 0.5)
    screen_time_before_sleep = st.slider("Screen Time Sebelum Tidur (jam)", 0.0, 3.0, 0.8, 0.1)
    breaks_during_work = st.slider("Jumlah Istirahat Selama Kerja", 0, 10, 3)
    coffee_consumption_per_day = st.slider("Konsumsi Kopi per Hari (gelas)", 0, 10, 2)
    days_feeling_burnout_per_month = st.slider("Hari Merasa Burnout per Bulan", 0, 31, 5)
    weekly_offline_hours = st.slider("Jam Offline per Minggu", 0.0, 40.0, 15.0, 0.5)
    job_satisfaction_score = st.slider("Skor Kepuasan Kerja (0-10)", 0.0, 10.0, 6.0, 0.1)
    uses_focus_apps = st.checkbox("Menggunakan Aplikasi Fokus?", value=False)
    has_digital_wellbeing_enabled = st.checkbox("Fitur Digital Wellbeing Aktif?", value=False)
    prediksi_btn = st.button("Prediksi Produktivitas", type="primary", use_container_width=True)

with col_hasil:
    st.subheader("Panel Hasil")
    if prediksi_btn:
        raw = {
            "age": age, "daily_social_media_time": daily_social_media_time,
            "number_of_notifications": number_of_notifications, "work_hours_per_day": work_hours_per_day,
            "perceived_productivity_score": perceived_productivity_score, "stress_level": stress_level,
            "sleep_hours": sleep_hours, "screen_time_before_sleep": screen_time_before_sleep,
            "breaks_during_work": breaks_during_work, "uses_focus_apps": int(uses_focus_apps),
            "has_digital_wellbeing_enabled": int(has_digital_wellbeing_enabled),
            "coffee_consumption_per_day": coffee_consumption_per_day,
            "days_feeling_burnout_per_month": days_feeling_burnout_per_month,
            "weekly_offline_hours": weekly_offline_hours, "job_satisfaction_score": job_satisfaction_score,
            "gender": gender, "job_type": job_type, "social_platform_preference": platform,
        }
        df_input = pd.DataFrame([raw])
        df_encoded = pd.get_dummies(df_input, columns=["gender", "job_type", "social_platform_preference"])
        df_encoded = df_encoded.reindex(columns=kolom_model, fill_value=0)

        pred = model.predict(df_encoded)[0]
        proba = model.predict_proba(df_encoded)[0][1]

        if pred == 1:
            st.error(f"Tidak Produktif (Probabilitas: {proba*100:.1f}%)")
        else:
            st.success(f"Produktif (Probabilitas Tidak Produktif: {proba*100:.1f}%)")
        st.metric("Probabilitas Tidak Produktif", f"{proba*100:.1f}%")
        st.progress(float(proba))
    else:
        st.info("Atur parameter di panel kiri lalu klik 'Prediksi Produktivitas'.")