import streamlit as st
import pandas as pd
import joblib

# =========================
# PATH
# =========================
ARTIFACT_PATH = "models/asthma_knn_artifact.pkl"
LOGO_PATH = "assets/logo_darmajaya.png"

st.set_page_config(
    page_title="Prediksi Risiko Asma (KNN)",
    page_icon="🫁",
    layout="wide"
)

# =========================
# LABEL & FITUR
# =========================
LABEL_ID = {
    "Age": "Usia",
    "Gender": "Jenis Kelamin",
    "BMI": "BMI (kg/m²)",
    "Smoking_Status": "Status Merokok",
    "Family_History": "Riwayat Keluarga Asma",
    "Allergies": "Alergi",
    "Air_Pollution_Level": "Tingkat Polusi Udara",
    "Physical_Activity_Level": "Tingkat Aktivitas Fisik",
    "Occupation_Type": "Jenis Pekerjaan",
    "Comorbidities": "Komorbid (Penyakit Penyerta)",
    "Medication_Adherence": "Kepatuhan Minum Obat",
    "Number_of_ER_Visits": "Jumlah Kunjungan IGD",
    "Peak_Expiratory_Flow": "Peak Expiratory Flow (PEF)",
    "FeNO_Level": "Kadar FeNO",
}

ALL_FEATURES = [
    "Age","Gender","BMI","Smoking_Status","Family_History","Allergies",
    "Air_Pollution_Level","Physical_Activity_Level","Occupation_Type",
    "Comorbidities","Medication_Adherence","Number_of_ER_Visits",
    "Peak_Expiratory_Flow","FeNO_Level",
]

NUMERIC_CFG = {
    "Age": dict(min_value=1, max_value=100, value=30, step=1),
    "BMI": dict(min_value=10.0, max_value=60.0, value=25.0, step=0.1),
    "Number_of_ER_Visits": dict(min_value=0, max_value=50, value=0, step=1),
    "Peak_Expiratory_Flow": dict(min_value=0.0, max_value=1000.0, value=300.0, step=1.0),
    "FeNO_Level": dict(min_value=0.0, max_value=500.0, value=25.0, step=1.0),
}

# =========================
# LOAD ARTIFACT
# =========================
@st.cache_resource(show_spinner=False)
def load_artifact(path):
    art = joblib.load(path)
    return art["model"], art.get("scaler"), art.get("encoders", {}), art["selected_features"]

try:
    model, scaler, encoders, selected_features = load_artifact(ARTIFACT_PATH)
except Exception as e:
    st.error(f"Gagal load artifact:\n\n{e}")
    st.stop()

# =========================
# HELPER
# =========================
def label(col):
    return LABEL_ID.get(col, col)

def input_widget(col):
    if col in encoders:
        options = list(encoders[col].classes_)
        return st.selectbox(label(col), options, index=0)
    if col in NUMERIC_CFG:
        return st.number_input(label(col), **NUMERIC_CFG[col])
    return st.number_input(label(col), value=0.0)

def preprocess(inputs):
    df = pd.DataFrame([inputs])

    for c, le in encoders.items():
        if c in df.columns:
            df[c] = le.transform(df[c].astype(str))

    df = df[ALL_FEATURES].apply(pd.to_numeric, errors="coerce")

    if df.isnull().any().any():
        raise ValueError("Ada input yang kosong / tidak valid.")

    # ✅ FIX UTAMA: scaling SEMUA fitur (sesuai saat training)
    if scaler is not None:
        df_scaled = pd.DataFrame(scaler.transform(df), columns=ALL_FEATURES)
    else:
        df_scaled = df.copy()

    # model hanya pakai fitur terpilih
    X = df_scaled[selected_features]
    return X

def risk_block(pred, prob):
    if pred == 1:
        st.error("⚠️ Risiko Tinggi (Positif Asma)")
    else:
        st.success("✅ Risiko Rendah (Negatif Asma)")

    if prob is not None:
        st.markdown(f"**Probabilitas Risiko:** `{prob*100:.2f}%`")
        st.progress(min(max(prob, 0.0), 1.0))

# =========================
# SIDEBAR
# =========================
st.sidebar.image(LOGO_PATH, use_container_width=True)
st.sidebar.title("🫁 Asthma Risk Prediction")
st.sidebar.caption("Institut Informatika dan Bisnis Darmajaya")
st.sidebar.markdown("---")
st.sidebar.write("**Algoritma:** KNN")
st.sidebar.write("**Preprocessing:** Label Encoding + MinMax Scaling")
st.sidebar.write("**Seleksi Fitur:** Chi-Square")

# =========================
# HEADER
# =========================
st.title("🫁 Dashboard Prediksi Risiko Penyakit Asma (KNN)")
st.markdown("**Eldrada Intan Putri**")
st.caption("Masukkan data pasien, lalu klik **Prediksi Sekarang**.")

# =========================
# METRIC
# =========================
m1, m2, m3 = st.columns(3)
m1.metric("Jumlah Variabel Input", len(ALL_FEATURES))
m2.metric("Fitur Dipakai Model", len(selected_features))
m3.metric("Status Model", "Loaded ✅")

# =========================
# FORM INPUT
# =========================
with st.form("form_asma"):
    c1, c2, c3 = st.columns(3)
    inputs = {}

    with c1:
        st.markdown("**Demografis & Kebiasaan**")
        for f in ["Age","Gender","BMI","Smoking_Status","Family_History"]:
            inputs[f] = input_widget(f)

    with c2:
        st.markdown("**Lingkungan & Riwayat**")
        for f in ["Allergies","Air_Pollution_Level","Physical_Activity_Level","Occupation_Type","Comorbidities"]:
            inputs[f] = input_widget(f)

    with c3:
        st.markdown("**Kondisi Klinis**")
        for f in ["Medication_Adherence","Number_of_ER_Visits","Peak_Expiratory_Flow","FeNO_Level"]:
            inputs[f] = input_widget(f)

    submit = st.form_submit_button("🔍 Prediksi Sekarang")

# =========================
# HASIL
# =========================
st.subheader("📊 Hasil Prediksi")

if submit:
    try:
        X = preprocess(inputs)
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None

        left, right = st.columns([1,1])
        with left:
            risk_block(pred, prob)
        with right:
            st.markdown("**Ringkasan Input**")
            st.dataframe(pd.DataFrame([inputs]).rename(columns=LABEL_ID), use_container_width=True)

        with st.expander("Fitur yang dipakai model"):
            st.write(selected_features)
            st.dataframe(X, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi error:\n\n{e}")

st.caption("© Eldrada Intan Putri — Institut Informatika dan Bisnis Darmajaya")
