# DiaPredict AI — Kelompok 10

Aplikasi prediksi diabetes menggunakan Machine Learning (Random Forest).

---

## Cara Menjalankan Notebook (Google Colab)

1. Buka file `diabetes_classification.ipynb` di Google Colab
2. Klik **Runtime → Run All**
3. Selesai, tidak perlu install apapun

Dataset sudah tersedia secara publik dan akan otomatis diunduh saat notebook dijalankan.

---

## Cara Menjalankan Aplikasi Streamlit (Lokal)

Install library yang dibutuhkan:

```
pip install streamlit scikit-learn pandas numpy joblib matplotlib seaborn xgboost imbalanced-learn
```

Jalankan aplikasi:

```
streamlit run app_streamlit.py
```

Buka browser di: http://localhost:8501

---

## File yang Dibutuhkan

- `app_streamlit.py` — aplikasi utama
- `random_forest_diabetes_best.pkl` — model ML
- `scaler.pkl` — normalisasi data
- `pca.pkl` — reduksi dimensi
- `diabetes_classification.ipynb` — notebook training

---

## Informasi Proyek

- **Nama Proyek**: DiaPredict AI — Klasifikasi Diabetes
- **Kelompok**: Kelompok 10
- **Model**: Random Forest
- **Dataset**: Diabetes Dataset (100.000 data)
- **Akurasi**: ~97%
