import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Diabetes Predictor AI", page_icon="💉", layout="centered")

# 2. Load semua komponen (Model, Scaler, PCA)
@st.cache_resource
def load_assets():
    model = joblib.load('random_forest_diabetes_best.pkl')
    scaler = joblib.load('scaler_diabetes.pkl')
    pca = joblib.load('pca_diabetes.pkl')
    selected_features = joblib.load('selected_features.pkl')
    return model, scaler, pca, selected_features

try:
    model, scaler, pca, selected_features = load_assets()
except Exception as e:
    st.error("Gagal memuat model. Pastikan file .pkl sudah ada di folder yang sama.")
    st.stop()

# 3. Header Aplikasi
st.title("💉 Diabetes Prediction AI")
st.markdown("""
Aplikasi ini menggunakan model **Random Forest** untuk memprediksi risiko diabetes berdasarkan data klinis pasien.
""")

# 4. Form Input User (Data Asli)
st.sidebar.header("Input Data Pasien")

age = st.sidebar.number_input("Usia (Age)", min_value=1, max_value=100, value=30)
gender = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"])
bmi = st.sidebar.number_input("BMI (Indeks Massa Tubuh)", min_value=10.0, max_value=50.0, value=25.0)
hba1c = st.sidebar.number_input("HbA1c Level", min_value=3.0, max_value=10.0, value=5.5)
glucose = st.sidebar.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)
hypertension = st.sidebar.selectbox("Riwayat Hipertensi?", [0, 1])
heart_disease = st.sidebar.selectbox("Riwayat Penyakit Jantung?", [0, 1])

# 5. Tombol Prediksi
if st.button("Analisis Risiko Diabetes"):
    
    # -- Proses Preprocessing (Harus sama dengan Notebook) --
    
    # Buat dummy data untuk semua kolom yang digunakan saat training (15 kolom asli)
    # Ini adalah simulasi sederhana, idealnya semua kolom diinput user
    data_dict = {
        'year': 2024,
        'gender': 1 if gender == "Male" else 0,
        'age': age,
        'location': 0, # Default/Median
        'race:AfricanAmerican': 0,
        'race:Asian': 0,
        'race:Caucasian': 1,
        'race:Hispanic': 0,
        'race:Other': 0,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': 0, # never
        'bmi': bmi,
        'hbA1c_level': hba1c,
        'blood_glucose_level': glucose
    }
    
    df_input = pd.DataFrame([data_dict])
    
    # a. Scaling
    # Ambil kolom numerik yang biasa di-scale
    numeric_cols = scaler.feature_names_in_
    df_input_scaled = df_input.copy()
    df_input_scaled[numeric_cols] = scaler.transform(df_input[numeric_cols])
    
    # b. PCA
    input_pca = pca.transform(df_input_scaled)
    df_pca = pd.DataFrame(input_pca, columns=[f'PC{i+1}' for i in range(input_pca.shape[1])])
    
    # c. Feature Selection
    X_final = df_pca[selected_features]
    
    # d. Predict
    prediction = model.predict(X_final)[0]
    probability = model.predict_proba(X_final)[0][1]
    
    # 6. Tampilkan Hasil
    st.subheader("Hasil Analisis:")
    if prediction == 1:
        st.error(f"⚠️ **TERDETEKSI RISIKO DIABETES**")
        st.write(f"Tingkat Keyakinan Model: **{probability:.2%}**")
        st.write("Saran: Segera konsultasikan hasil ini dengan tenaga medis profesional.")
    else:
        st.success(f"✅ **HASIL PREDIKSI: SEHAT**")
        st.write(f"Tingkat Keyakinan Model: **{1-probability:.2%}**")
        st.write("Tetap jaga pola makan dan gaya hidup sehat!")

st.info("Catatan: Aplikasi ini hanya untuk tujuan edukasi dan demonstrasi proyek ML.")
