import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

st.set_page_config(page_title="DiaPredict AI Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #FAFBFC;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Hide top header bar in sidebar */
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* Primary buttons */
    .stButton > button {
        background-color: #0E429B;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #0C3882;
        color: white;
    }
    
    /* Outline buttons */
    .outline-btn > .stButton > button {
        background-color: white;
        color: #1E293B;
        border: 1px solid #CBD5E1;
    }
    .outline-btn > .stButton > button:hover {
        border-color: #94A3B8;
        color: #1E293B;
    }

    /* Cards */
    .custom-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    
    /* Blue tinted card */
    .blue-card {
        background-color: #F0F4FA;
        border: 1px solid #D6E4F5;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    /* Dark Blue card for Precision Check */
    .dark-card {
        background-color: #0F3E90;
        color: white;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
    
    /* Stepper */
    .stepper-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
        font-size: 14px;
        color: #64748B;
        font-weight: 500;
    }
    .step-circle {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background-color: #E2E8F0;
        color: #64748B;
        font-size: 12px;
    }
    .step-circle.active {
        background-color: #0E429B;
        color: white;
    }
    .step-line {
        height: 1px;
        width: 40px;
        background-color: #E2E8F0;
    }
    .step-text.active {
        color: #0E429B;
    }
    
    /* Upload boxes */
    .upload-box {
        border: 1px dashed #CBD5E1;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        background-color: #FFFFFF;
        margin-bottom: 10px;
    }
    .upload-box h4 {
        margin: 5px 0 0 0;
        font-size: 14px;
        color: #1E293B;
    }
    .upload-box p {
        margin: 0;
        font-size: 12px;
        color: #64748B;
    }

    /* Form labels */
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: #475569;
        font-weight: 500;
    }

    /* Sub-text for inputs */
    .input-subtext {
        font-size: 11px;
        color: #94A3B8;
        margin-top: -10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


FEATURE_COLS = [
    'year', 'gender', 'age',
    'race:AfricanAmerican', 'race:Asian', 'race:Caucasian',
    'race:Hispanic', 'race:Other',
    'hypertension', 'heart_disease', 'smoking_history',
    'bmi', 'hbA1c_level', 'blood_glucose_level'
]

def find_model_files():
    files = []
    candidates = [
        'random_forest_diabetes_best.pkl',
        'random_forest_diabetes_tuned.pkl',
        'random_forest_diabetes.pkl'
    ]
    for f in candidates:
        if os.path.exists(f):
            files.append(f)
    return files

def load_pickle(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return None

def preprocess_input(df, label_encoders=None, scaler=None, pca=None):
    df_proc = df.copy()
    if label_encoders is not None:
        for col, enc in label_encoders.items():
            if col in df_proc.columns:
                df_proc[col] = enc.transform(df_proc[col].astype(str))
    else:
        if 'gender' in df_proc.columns:
            df_proc['gender'] = df_proc['gender'].map({'Female':0,'Male':1}).fillna(0)
        if 'smoking_history' in df_proc.columns:
            df_proc['smoking_history'] = df_proc['smoking_history'].astype(str)
            df_proc['smoking_history'] = pd.factorize(df_proc['smoking_history'])[0]

    if any(c.startswith('race:') for c in FEATURE_COLS):
        race_cols = [c for c in FEATURE_COLS if c.startswith('race:')]
        for rc in race_cols:
            if rc not in df_proc.columns:
                df_proc[rc] = 0

    X = df_proc[FEATURE_COLS]
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS)
    if pca is not None:
        X = pca.transform(X)
    return X


def render_sidebar():
    st.sidebar.markdown("""
        <h2 style='color:#0E429B; margin-bottom: 0;'>DiaPredict AI</h2>
        <p style='color:#64748B; font-size: 14px; margin-top: -5px;'>Dasbor Klinis</p>
        <br/>
    """, unsafe_allow_html=True)
    
    # Navigasi kustom
    st.sidebar.markdown("""
        <div style='background-color:#1652C4; color:white; padding:10px 15px; border-radius:6px; margin-bottom:30px; font-weight:500; display:flex; align-items:center;'>
            📄 &nbsp; Prediktor
        </div>
    """, unsafe_allow_html=True)

def main():
    render_sidebar()
    
    # Load default models
    model_files = find_model_files()
    model_choice = model_files[0] if model_files else None
    model = load_pickle(model_choice) if model_choice else None
    
    scaler_file = 'scaler.pkl' if os.path.exists('scaler.pkl') else None
    scaler = load_pickle(scaler_file) if scaler_file else None
    
    pca_file = 'pca.pkl' if os.path.exists('pca.pkl') else None
    pca = load_pickle(pca_file) if pca_file else None

    # Main Layout
    main_col, right_col = st.columns([2.2, 1])
    
    with main_col:
        st.markdown("<h1 style='color:#0F172A; margin-bottom:5px;'>Prediksi Sampel Tunggal</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748B; font-size:15px; margin-bottom:30px; line-height:1.5;'>Masukkan data klinis pasien di bawah ini untuk menghasilkan penilaian risiko diabetes secara real-time berdasarkan model RF yang telah dilatih.</p>", unsafe_allow_html=True)
        

        
        with st.form("prediction_form"):
            st.markdown("<h3 style='color:#1E293B; margin-top:20px; font-size:20px;'>🪪 Metadata Klinis</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                year = st.number_input('Tahun', value=2023, step=1)
                age = st.number_input('Usia', min_value=0, max_value=120, value=21)
            with col2:
                gender = st.selectbox('Jenis Kelamin', options=['Male', 'Female'], format_func=lambda x: 'Laki-laki' if x == 'Male' else 'Perempuan')
                race = st.selectbox('Ras', options=['race:Asian', 'race:AfricanAmerican', 'race:Caucasian', 'race:Hispanic', 'race:Other'],
                                    format_func=lambda x: {
                                        'race:Asian': 'Asia',
                                        'race:AfricanAmerican': 'Afrika-Amerika',
                                        'race:Caucasian': 'Kaukasia',
                                        'race:Hispanic': 'Hispanik',
                                        'race:Other': 'Lainnya'
                                    }.get(x, x))
                
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:#1E293B; margin-top:10px; font-size:20px;'>🏥 Riwayat Medis</h3>", unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            with col3:
                hypertension_disp = st.selectbox('Hipertensi', options=['Tidak', 'Ya'])
                hypertension = 1 if hypertension_disp == 'Ya' else 0
            with col4:
                heart_disease_disp = st.selectbox('Penyakit Jantung', options=['Tidak', 'Ya'])
                heart_disease = 1 if heart_disease_disp == 'Ya' else 0
            smoking_history = st.selectbox('Riwayat Merokok', options=['never','current','former','not current','ever'],
                                           format_func=lambda x: {
                                               'never': 'Tidak Pernah',
                                               'current': 'Saat Ini Merokok',
                                               'former': 'Mantan Perokok',
                                               'not current': 'Tidak Saat Ini',
                                               'ever': 'Pernah'
                                           }.get(x, x))
            
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:#1E293B; margin-top:10px; font-size:20px;'>📈 Data Biometrik</h3>", unsafe_allow_html=True)
            
            # --- Kalkulator BMI ---
            st.markdown("""
            <div style='background-color:#F0F4FA; border:1px dashed #93C5FD; border-radius:10px; padding:15px 20px; margin-bottom:15px;'>
                <p style='margin:0; font-size:14px; font-weight:600; color:#1E3A5F;'>🔢 Tidak tahu BMI Anda? Gunakan kalkulator di bawah ini:</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📐 Buka Kalkulator BMI", expanded=False):
                st.markdown("<p style='font-size:13px; color:#64748B; margin-bottom:10px;'>Masukkan tinggi dan berat badan Anda, BMI akan dihitung otomatis.</p>", unsafe_allow_html=True)
                kalk_col1, kalk_col2 = st.columns(2)
                with kalk_col1:
                    tinggi_cm = st.number_input('Tinggi Badan (cm)', min_value=50.0, max_value=250.0, value=165.0, step=0.5, format='%.1f', key='tinggi_bmi')
                with kalk_col2:
                    berat_kg = st.number_input('Berat Badan (kg)', min_value=10.0, max_value=300.0, value=56.0, step=0.5, format='%.1f', key='berat_bmi')
                
                bmi_kalkulasi = berat_kg / ((tinggi_cm / 100) ** 2)
                
                # Kategori BMI
                if bmi_kalkulasi < 18.5:
                    kat_warna = "#3B82F6"
                    kat_label = "Kurus (Underweight)"
                elif bmi_kalkulasi < 25.0:
                    kat_warna = "#22C55E"
                    kat_label = "Normal"
                elif bmi_kalkulasi < 30.0:
                    kat_warna = "#F59E0B"
                    kat_label = "Gemuk (Overweight)"
                else:
                    kat_warna = "#EF4444"
                    kat_label = "Obesitas"
                
                st.markdown(f"""
                <div style='background-color:#F8FAFC; border-radius:8px; padding:15px; text-align:center; border: 1px solid #E2E8F0; margin-top:10px;'>
                    <p style='margin:0; font-size:13px; color:#64748B;'>Hasil BMI Anda</p>
                    <h2 style='margin:5px 0; font-size:36px; color:{kat_warna}; font-weight:700;'>{bmi_kalkulasi:.2f}</h2>
                    <span style='background-color:{kat_warna}20; color:{kat_warna}; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600;'>{kat_label}</span>
                    <p style='margin-top:10px; font-size:12px; color:#94A3B8;'>Nilai ini akan otomatis digunakan pada field BMI di bawah</p>
                </div>
                """, unsafe_allow_html=True)
            
            col5, col6, col7 = st.columns(3)
            with col5:
                bmi = st.number_input('BMI', value=round(bmi_kalkulasi, 2) if 'bmi_kalkulasi' in dir() else 20.80, format='%.2f')
                st.markdown("<div class='input-subtext'>Rentang normal: 18.5 - 24.9</div>", unsafe_allow_html=True)
            with col6:
                hbA1c_level = st.number_input('Kadar HbA1c', value=5.60, format='%.2f')
                st.markdown("<div class='input-subtext'>Diukur dalam %</div>", unsafe_allow_html=True)
            with col7:
                blood_glucose_level = st.number_input('Kadar Glukosa Darah', value=180.00, format='%.2f')
                st.markdown("<div class='input-subtext'>mg/dL (puasa)</div>", unsafe_allow_html=True)
            

            st.markdown("<br/>", unsafe_allow_html=True)
            submit_col, clear_col, _ = st.columns([1.5, 1, 1.5])
            with submit_col:
                submitted = st.form_submit_button('📊 Analisis Prediksi')
            with clear_col:
                st.markdown("<div class='outline-btn'>", unsafe_allow_html=True)
                st.form_submit_button('Bersihkan Form')
                st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        # Tampilan Prediksi Risiko
        if not submitted:
            st.markdown("""
            <div class='blue-card' style='text-align:center;'>
                <h4 style='color:#1E293B; text-align:left; margin-top:0;'>Prediksi Risiko</h4>
                <div style='background-color:#E2E8F0; width:50px; height:50px; border-radius:12px; display:inline-flex; align-items:center; justify-content:center; margin: 20px 0;'>
                    <span style='font-size:24px; color:#0E429B;'>📈</span>
                </div>
                <p style='color:#475569; font-size:14px; margin-bottom:10px;'>Isi data klinis pasien untuk menghasilkan skor risiko dan laporan lengkap.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if model is None:
                st.error("Model .pkl tidak ditemukan. Silakan upload model di bagian bawah.")
            else:
                sample = pd.DataFrame([{
                    'year': year,
                    'gender': gender,
                    'age': age,
                    'race:AfricanAmerican': 1 if race=='race:AfricanAmerican' else 0,
                    'race:Asian': 1 if race=='race:Asian' else 0,
                    'race:Caucasian': 1 if race=='race:Caucasian' else 0,
                    'race:Hispanic': 1 if race=='race:Hispanic' else 0,
                    'race:Other': 1 if race=='race:Other' else 0,
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                    'smoking_history': smoking_history,
                    'bmi': bmi,
                    'hbA1c_level': hbA1c_level,
                    'blood_glucose_level': blood_glucose_level
                }])
                X = preprocess_input(sample, label_encoders=None, scaler=scaler, pca=pca)
                try:
                    pred = model.predict(X)
                    prob = model.predict_proba(X)[:,1]
                    
                    bg_col = "#FEE2E2" if pred[0] == 1 else "#DCFCE7"
                    text_col = "#991B1B" if pred[0] == 1 else "#166534"
                    status = "Risiko Tinggi (Diabetes)" if pred[0] == 1 else "Risiko Rendah (Sehat)"
                    
                    st.markdown(f"""
                    <div class='blue-card' style='background-color:{bg_col}; border-color:{text_col};'>
                        <h4 style='color:{text_col}; margin-top:0;'>Prediksi Risiko</h4>
                        <h2 style='color:{text_col}; font-size:24px; margin-top:10px;'>{status}</h2>
                        <p style='color:{text_col}; font-size:14px;'>Probabilitas: <b>{prob[0]:.1%}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f'Gagal melakukan prediksi: {e}')




if __name__ == '__main__':
    main()
