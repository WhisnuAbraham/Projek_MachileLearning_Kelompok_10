pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost

import os
import warnings
import joblib
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    roc_curve
)
from imblearn.over_sampling import SMOTE
from IPython.display import display
from google.colab import drive

warnings.filterwarnings('ignore')
print("Semua library berhasil diimport!")

drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/PA 3 KEL 10-20260406T061058Z-3-001/PA 3 KEL 10/ML/diabetes_dataset.csv')
print("Dataset berhasil dimuat")
print(f"Jumlah baris: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")
df.head() 

print("INFORMASI STRUKTUR DATASET")
df.info()
print("\nSTATISTIK DESKRIPTIF")
print(df.describe())
print("\nJUMLAH MISSING VALUE")
print(df.isnull().sum())
print(f"\nTotal keseluruhan missing value di dataset: {df.isnull().sum().sum()}")
print("\nDISTRIBUSI LABEL (KOLOM 'diabetes')")
print(df['diabetes'].value_counts())
print("\nPersentase Distribusi Label:")
print(df['diabetes'].value_counts(normalize=True) * 100)

print(df.isnull().sum())

print("PENGECEKAN JUMLAH OUTLIER (METODE IQR)")
outlier_cols = []
numeric_cols = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level']

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    #jumlah data outlier
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    n_outliers = len(outliers)
    pct_outliers = (n_outliers / len(df)) * 100
    
    print(f"Kolom '{col}': {n_outliers} baris outlier ({pct_outliers:.2f}%)")
    
    if n_outliers > 0:
        outlier_cols.append(col)

print("\nVISUALISASI BOXPLOT KOLOM DENGAN OUTLIER")
if outlier_cols:
    plt.figure(figsize=(14, 10))
    for i, col in enumerate(outlier_cols, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(f"Boxplot of {col} (Menampilkan Outlier)", fontsize=12)
        plt.xlabel(col, fontsize=10)
    plt.tight_layout()
    plt.show()
else:
    print("Tidak ditemukan kolom numerik yang memiliki outlier.")


print("DISTRIBUSI LABEL diabetes")
counts = df['diabetes'].value_counts()
print(counts)
print("\nPersentase:")
print(df['diabetes'].value_counts(normalize=True) * 100)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.pie(counts, labels=['Tidak Diabetes (0)', 'Diabetes (1)'], autopct='%1.1f%%', 
        startangle=140, colors=['#66b3ff', '#ff9999'], explode=(0, 0.1), shadow=True)
plt.title("Persentase Distribusi Kelas", fontsize=14, pad=15)

plt.subplot(1, 2, 2)
ax = sns.countplot(x='diabetes', data=df, palette=['#66b3ff', '#ff9999'])
plt.title("Jumlah Pasien per Kelas", fontsize=14, pad=15)
plt.xlabel("Status Diabetes (0 = Tidak, 1 = Ya)", fontsize=12)
plt.ylabel("Jumlah Pasien", fontsize=12)
plt.xticks([0, 1], ['Tidak Diabetes (0)', 'Diabetes (1)'])

for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.show()


print("KATEGORI RIWAYAT MEROKOK ('smoking_history')")
print("Kategori Unik:", df['smoking_history'].unique())
print("\nJumlah Pasien per Kategori:")
print(df['smoking_history'].value_counts())

print("\nEKSPLORASI PASIEN ANAK (USIA < 10 TAHUN)")
df_kids = df[df['age'] < 10]
print(f"Total pasien anak di bawah 10 tahun: {len(df_kids)} baris ({len(df_kids)/len(df)*100:.2f}% dari total dataset)")
display(df_kids.head())
plt.figure(figsize=(16, 6))

#Bar Chart Riwayat Merokok Keseluruhan
plt.subplot(1, 2, 1)
ax1 = sns.countplot(y='smoking_history', data=df, order=df['smoking_history'].value_counts().index, palette='viridis')
plt.title("Distribusi Riwayat Merokok", fontsize=14, pad=15)
plt.xlabel("Jumlah Pasien", fontsize=12)
plt.ylabel("Kategori Merokok", fontsize=12)

for p in ax1.patches:
    ax1.annotate(f'{int(p.get_width()):,}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                 ha='left', va='center', fontsize=11, color='black', xytext=(5, 0),
                 textcoords='offset points')

#Distribusi Usia Pasien Anak (< 10 Tahun)
plt.subplot(1, 2, 2)
sns.histplot(df_kids['age'], bins=10, kde=True, color='#ff9999', edgecolor='black')
plt.title("Distribusi Usia Pasien Anak (< 10 Tahun)", fontsize=14, pad=15)
plt.xlabel("Usia (Tahun)", fontsize=12)
plt.ylabel("Jumlah Pasien Anak", fontsize=12)

plt.tight_layout()
plt.show()


print("DATA NOISE PADA KOLOM 'gender'")
noise_gender = df[df['gender'] == 'Other']
print(f"Jumlah baris dengan gender 'Other': {len(noise_gender)}")
print("\nDetail baris noise:")
print(noise_gender[['gender', 'age', 'smoking_history', 'diabetes']].to_string())
mode_gender = df[df['gender'] != 'Other']['gender'].mode()[0]
print(f"\nModus gender (yang akan digunakan untuk imputasi): '{mode_gender}'")
df['gender'] = df['gender'].replace('Other', mode_gender)
remaining = (df['gender'] == 'Other').sum()
print(f"\nJumlah 'Other' setelah imputasi : {remaining}")
print(f"Status Imputasi                 : {'BERHASIL' if remaining == 0 else 'GAGAL'}")
print("\nDistribusi gender setelah imputasi:")
print(df['gender'].value_counts())

no_info_count = (df['smoking_history'] == 'No Info').sum()

mode_value = df[df['smoking_history'] != 'No Info']['smoking_history'].mode()[0]

print(f"PROSES IMPUTASI KOLOM 'smoking_history")
print(f"Jumlah baris 'No Info' sebelum imputasi : {no_info_count}")
print(f"Modus yang digunakan : '{mode_value}'")

# Melakukan penggantian (imputasi)
df['smoking_history'] = df['smoking_history'].replace('No Info', mode_value)

#Verifikasi setelah imputasi
no_info_remaining = (df['smoking_history'] == 'No Info').sum()
print(f"Jumlah baris 'No Info' setelah imputasi   : {no_info_remaining}")


# bar chart dari hasil value_counts
df['smoking_history'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Distribusi Riwayat Merokok')
plt.xlabel('Kategori Riwayat Merokok')
plt.ylabel('Jumlah Pasien/Data')
plt.xticks(rotation=45)

plt.show()

def batasi_nilai_iqr(data, column, min_val=None, max_val=None):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    if min_val is not None:
        lower = max(lower, min_val)
    if max_val is not None:
        upper = min(upper, max_val)

    jumlah_outlier = ((data[column] < lower) | (data[column] > upper)).sum()
    data[column] = data[column].clip(lower=lower, upper=upper)

    print(f" Kolom '{column}'")
    print(f" Batas bawah : {lower:.2f} | Batas atas : {upper:.2f}")
    print(f" Nilai ekstrem yang dibatasi : {jumlah_outlier} baris")
    return data

print("PENANGANAN OUTLIER (PEMBATASAN NILAI / IQR)")
print(f"Jumlah baris sebelum : {len(df)}")
print("-" * 52)

df = batasi_nilai_iqr(df, 'age', min_val=0)
print("-" * 52)

df = batasi_nilai_iqr(df, 'bmi', min_val=10)
print("-" * 52)

df = batasi_nilai_iqr(df, 'hbA1c_level')
print("-" * 52)
df = batasi_nilai_iqr(df, 'blood_glucose_level')
print("-" * 52)

print(f"Jumlah baris sesudah : {len(df)}")

# Hapus kolom 'location' dari preprocessing
df = df.drop(columns=['location'])
print("Kolom 'location' dihapus dari data.")

print("Status : SELESAI - Nilai ekstrem berhasil dibatasi")


# Encode kolom kategorikal ke angka
df_encoded = df.copy()
le = LabelEncoder()
kolom_kategorikal = df_encoded.select_dtypes(include=['object']).columns
print("Kolom yang di-encode:", list(kolom_kategorikal))
for col in kolom_kategorikal:
    df_encoded[col] = le.fit_transform(df_encoded[col])
print("Tipe data setelah encoding:")
print(df_encoded.dtypes)
X = df_encoded.drop(columns=['diabetes'])
y = df_encoded['diabetes']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# distribusi sebelum SMOTE
print("Sebelum SMOTE (data training):")
print(y_train.value_counts())
rasio = y_train.value_counts()[0] / y_train.value_counts()[1]
print("Rasio imbalance : {:.1f}x".format(rasio))
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# distribusi sesudah SMOTE
print("\nSesudah SMOTE (data training):")
print(pd.Series(y_train_smote).value_counts())
print("Status : SELESAI - Kelas sudah seimbang")


print(df_encoded.dtypes)

X = df_encoded.drop(columns=['diabetes'])
y = df_encoded['diabetes']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y    
)
# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  
# PCA (Dimensionality Reduction)
do_pca = True 
if do_pca:
    pca = PCA(n_components=0.95, random_state=42) 
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)  
    
    print(f"PCA Selesai: Fitur berkurang dari {X_train.shape[1]} kolom menjadi {X_train_pca.shape[1]} kolom (Principal Components)")
else:
    X_train_pca = X_train_scaled
    X_test_pca = X_test_scaled
# SMOTE
smote = SMOTE(random_state=42)
X_train_final, y_train_final = smote.fit_resample(X_train_pca, y_train)
print("\nDISTRIBUSI KELAS")
print("Sesudah SMOTE (y_train_final):")
print(y_train_final.value_counts())
print("\nStatus: SELESAI ")

print("MEMULAI PELATIHAN MODEL RANDOM FOREST")
# Inisialisasi Model
rf = RandomForestClassifier(random_state=42)
# Latih model menggunakan data training yang SUDAH di-SMOTE
rf.fit(X_train_final, y_train_final)
y_pred_rf = rf.predict(X_test_pca)
# Evaluasi Hasil
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_rf))
print('\nClassification Report:')
print(classification_report(y_test, y_pred_rf))
print('Akurasi:', accuracy_score(y_test, y_pred_rf))

print("PELATIHAN MODEL RANDOM FOREST")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_final, y_train_final)

y_pred = rf.predict(X_test_pca)

# Evaluasi Matrix
print('Confusion Matrix (Threshold 0.5):')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report (Threshold 0.5):')
print(classification_report(y_test, y_pred))
print(f'Akurasi : {accuracy_score(y_test, y_pred):.4f}')
print(f'ROC AUC : {roc_auc_score(y_test, rf.predict_proba(X_test_pca)[:,1]):.4f}')

# TUNING THRESHOLD PREDIKSI 
y_proba = rf.predict_proba(X_test_pca)[:,1]
best_recall = 0
best_thresh = 0.5

for thresh in np.arange(0.1, 0.9, 0.01):
    y_pred_thresh = (y_proba >= thresh).astype(int)
    recall_1 = recall_score(y_test, y_pred_thresh, pos_label=1)
    if recall_1 > best_recall:
        best_recall = recall_1
        best_thresh = thresh

print(f'Threshold terbaik untuk mendeteksi Diabetes (Kelas 1): {best_thresh:.2f} (Recall = {best_recall:.2f})')
y_pred_best = (y_proba >= best_thresh).astype(int)
print(f'\nClassification Report, Threshold {best_thresh:.2f}):')
print(classification_report(y_test, y_pred_best))


print("ANALISIS FEATURE IMPORTANCE")
importances = rf.feature_importances_
feature_names = [f'PC{i+1}' for i in range(len(importances))]
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print('Skor Kepentingan Fitur (Principal Components):')
print(feat_imp.to_string())
plt.figure(figsize=(12, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title('Tingkat Kepentingan Komponen PCA - Random Forest', fontsize=14)
plt.xlabel('Importance Score')
plt.ylabel('Principal Component')
plt.tight_layout()
plt.show()
plt.show()



print("MODEL BASELINE (Random Forest - Sebelum Tuning)")
y_pred_baseline = rf.predict(X_test_pca)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_baseline))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_baseline))
print(f"Akurasi  : {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, rf.predict_proba(X_test_pca)[:,1]):.4f}")
cv_scores = cross_val_score(rf, X_train_final, y_train_final, cv=3, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


print("HYPERPARAMETER TUNING (Metode: GridSearchCV)")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf_grid = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_grid, 
    param_grid=param_grid, 
    cv=3, 
    scoring='f1',    
    n_jobs=-1,      
    verbose=1
)
print("Mulai melatih\n")
grid_search.fit(X_train_final, y_train_final)
print("\n" + "="*60)
print("HASIL TUNING :")
print("="*60)
print(f"Kombinasi Parameter Terbaik : {grid_search.best_params_}")
print(f"F1-Score Terbaik            : {grid_search.best_score_:.4f}\n")
best_rf_model = grid_search.best_estimator_


print("TUNED MODEL PERFORMANCE")
best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test_pca)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
print(f"Test Accuracy (Tuned): {tuned_accuracy:.4f}")
print(f"\nConfusion Matrix (Tuned):")
print(confusion_matrix(y_test, y_pred_tuned))
print(f"\nClassification Report (Tuned):")
print(classification_report(y_test, y_pred_tuned))
print("\n" + "="*60)
print("PERBANDINGAN: BASELINE vs TUNED MODEL")
print("="*60)
y_pred_baseline = rf.predict(X_test_pca)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
improvement = ((tuned_accuracy - baseline_accuracy) / baseline_accuracy) * 100
print(f"Baseline Accuracy :  {baseline_accuracy:.4f}")
print(f"Tuned Accuracy    :  {tuned_accuracy:.4f}")
print(f"Improvement       :  {improvement:+.2f}%")
print("="*60)

print("     ANALISIS AKHIR & PENYIMPANAN MODEL     ")
misclassified_idx = np.where(y_test != y_pred_tuned)[0]
print(f'Jumlah salah tebak: {len(misclassified_idx)} dari {len(y_test)} pasien ujian')
print('\nContoh 5 pasien yang salah ditebak oleh AI:')
display(pd.DataFrame({
    'Diagnosis Asli Dokter': y_test.values[misclassified_idx], 
    'Tebakan AI': y_pred_tuned[misclassified_idx]
}).head())

cm = confusion_matrix(y_test, y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Tuned Model)')
plt.show()
 
if hasattr(best_rf, 'feature_importances_'):
    feature_names = [f'PC{i+1}' for i in range(len(best_rf.feature_importances_))]
    feat_imp = pd.Series(best_rf.feature_importances_, index=feature_names)
    feat_imp.sort_values(ascending=True).plot(kind='barh', figsize=(10, 6), colormap='viridis')
    plt.title('Faktor Paling Mempengaruhi Diabetes (Tuned Model)')
    plt.xlabel('Tingkat Kepentingan (Importance Score)')
    plt.show()
else:
    print('Model tidak memiliki atribut feature_importances_.')
 
model_path = os.path.join(os.getcwd(), 'random_forest_diabetes_best.pkl')
joblib.dump(best_rf, model_path)
scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')
pca_path = os.path.join(os.getcwd(), 'pca.pkl')

joblib.dump(scaler, scaler_path)
joblib.dump(pca, pca_path)
print(f'SUCCESS! PCA disimpan di:\n{pca_path}')
print(f'SUCCESS! Model disimpan di:\n{model_path}')
print(f'SUCCESS! Scaler disimpan di:\n{scaler_path}')

print("Menyimpan data uji ke folder")
try:
    if 'X_test_pca' in globals() and 'y_test' in globals():
        joblib.dump(X_test_pca, 'X_test_pca.pkl')
        joblib.dump(y_test,     'y_test.pkl')
        if 'X_test' in globals():
            joblib.dump(X_test, 'X_test.pkl')
        print("Data uji berhasil disimpan (X_test_pca.pkl, y_test.pkl, X_test.pkl)")
    else:
        print("X_test_pca atau y_test tidak ditemukan di memori, dilewati.")
except Exception as e:
    print(f"Gagal menyimpan data uji: {e}")

try:
    if 'df' in globals():
        from sklearn.preprocessing import LabelEncoder
        label_encoders = {}
        for col in ['gender', 'smoking_history']:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            label_encoders[col] = le
        joblib.dump(label_encoders, 'label_encoders.pkl')
        print("Label encoders berhasil disimpan (label_encoders.pkl)")
    else:
        print("df tidak ditemukan di memori, pembuatan label_encoders.pkl dilewati.")
except Exception as e:
    print(f"Gagal menyimpan label encoders: {e}")

found = {}
for f in ['random_forest_diabetes_best.pkl', 'random_forest_diabetes_tuned.pkl', 'random_forest_diabetes.pkl']:
    if os.path.exists(f):
        found['model'] = f
        break
if 'model_path' in globals() and os.path.exists(model_path):
    found['model'] = model_path

if os.path.exists('scaler.pkl'):
    found['scaler'] = 'scaler.pkl'
elif 'scaler_path' in globals() and os.path.exists(scaler_path):
    found['scaler'] = scaler_path

if os.path.exists('pca.pkl'):
    found['pca'] = 'pca.pkl'
elif 'pca_path' in globals() and os.path.exists(pca_path):
    found['pca'] = pca_path

if os.path.exists('label_encoders.pkl'):
    found['label_encoders'] = 'label_encoders.pkl'

for test_file in ['X_test_pca.pkl', 'y_test.pkl', 'X_test.pkl']:
    if os.path.exists(test_file):
        found[test_file.split('.')[0]] = test_file

print('\nFiles detected to copy:')
for k, v in found.items():
    print(f" - {k}: {os.path.abspath(v)}")

# MENYALIN FILE KE GOOGLE DRIVE
drive_dest = None
if os.path.exists('/content/drive'):
    drive_dest = '/content/drive/MyDrive/diabetes_models'
else:
    candidates = [
        os.path.expanduser('~/Google Drive'),
        os.path.expanduser('~/Drive'),
        os.path.expanduser('~/My Drive'),
        os.path.expanduser('~/GoogleDrive')
    ]
    for p in candidates:
        if os.path.exists(p):
            drive_dest = os.path.join(p, 'diabetes_models')
            break

if drive_dest:
    os.makedirs(drive_dest, exist_ok=True)
    for name, path in found.items():
        try:
            dest = os.path.join(drive_dest, os.path.basename(path))
            shutil.copy2(path, dest)
            print(f"Copied {os.path.basename(path)} to {dest}")
        except Exception as e:
            print(f"Failed to copy {path} to drive: {e}")
    print('Done copying to Drive.')
else:
    print('No Google Drive folder detected on this system. If you want to save to Drive, mount it (Colab) or provide a Drive path.')


print("1. XGBOOST BASELINE MODEL (Default Parameters)")

X_train_xgb = X_train.copy()
X_test_xgb = X_test.copy()

xgb_baseline = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=0,
    scale_pos_weight=1  
)

xgb_baseline.fit(X_train_xgb, y_train)

y_pred_xgb_baseline = xgb_baseline.predict(X_test_xgb)
xgb_baseline_accuracy = accuracy_score(y_test, y_pred_xgb_baseline)
xgb_baseline_cv = cross_val_score(xgb_baseline, X_train_xgb, y_train, cv=3, scoring='accuracy', n_jobs=-1)

print(f"Test Accuracy (XGBoost Baseline): {xgb_baseline_accuracy:.4f}")
print(f"CV Accuracy (XGBoost Baseline): {xgb_baseline_cv.mean():.4f} (+/- {xgb_baseline_cv.std():.4f})")
print(f"ROC AUC Score: {roc_auc_score(y_test, xgb_baseline.predict_proba(X_test_xgb)[:,1]):.4f}")
print(f"\nClassification Report (XGBoost Baseline):")
print(classification_report(y_test, y_pred_xgb_baseline))
print()


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

print("XGBOOST OPTIMIZED - MENGGUNAKAN RANDOMIZEDSEARCHCV")
print("="*70)

X_train_xgb_opt = X_train.copy()
X_test_xgb_opt = X_test.copy()

print("\nXGBoost Baseline (Quick Reference)")
print("-" * 70)

xgb_baseline_opt = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1,
    verbosity=0
)

xgb_baseline_opt.fit(X_train_xgb_opt, y_train)
y_pred_xgb_baseline_opt = xgb_baseline_opt.predict(X_test_xgb_opt)
xgb_baseline_accuracy_opt = accuracy_score(y_test, y_pred_xgb_baseline_opt)

print(f"Baseline Test Accuracy: {xgb_baseline_accuracy_opt:.4f}")

print("\nHyperparameter Tuning dengan RandomizedSearchCV")
print("-" * 70)

xgb_param_dist_opt = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
}

xgb_model_opt = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1,
    verbosity=0
)

xgb_random_search_opt = RandomizedSearchCV(
    estimator=xgb_model_opt,
    param_distributions=xgb_param_dist_opt,
    n_iter=20,  
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

print("Training RandomizedSearchCV (20 iterasi, 60 model fits)...\n")
xgb_random_search_opt.fit(X_train_xgb_opt, y_train)

print(f"\nBest Parameters: {xgb_random_search_opt.best_params_}")
print(f"Best CV Score (F1): {xgb_random_search_opt.best_score_:.4f}")

print("\nXGBoost Tuned Model (Setelah Hyperparameter Tuning)")
print("-" * 70)

best_xgb_opt = xgb_random_search_opt.best_estimator_
y_pred_xgb_tuned_opt = best_xgb_opt.predict(X_test_xgb_opt)
xgb_tuned_accuracy_opt = accuracy_score(y_test, y_pred_xgb_tuned_opt)
xgb_tuned_cv_opt = cross_val_score(best_xgb_opt, X_train_xgb_opt, y_train, cv=3, scoring='accuracy', n_jobs=-1)

print(f"Test Accuracy (Tuned): {xgb_tuned_accuracy_opt:.4f}")
print(f"CV Score: {xgb_tuned_cv_opt.mean():.4f} (+/- {xgb_tuned_cv_opt.std():.4f})")
print(f"ROC AUC: {roc_auc_score(y_test, best_xgb_opt.predict_proba(X_test_xgb_opt)[:,1]):.4f}")

print(f"\nConfusion Matrix:")
cm_xgb_opt = confusion_matrix(y_test, y_pred_xgb_tuned_opt)
print(cm_xgb_opt)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_xgb_tuned_opt))

improvement_opt = ((xgb_tuned_accuracy_opt - xgb_baseline_accuracy_opt) / xgb_baseline_accuracy_opt) * 100
print(f"\n Improvement: {improvement_opt:+.2f}%")


print("RANDOM FOREST VS XGBOOST OPTIMIZED")

# Pastikan data tersedia
try:
    X_train
    X_test
    y_train
    y_test
except NameError:
    raise NameError("Data X_train, X_test, y_train, atau y_test tidak ditemukan di environment. "
                    "Pastikan sel preprocessing data telah dijalankan.")

if 'X_selected_test' in globals() and 'X_selected_train' in globals():
    X_test_rf = X_selected_test
    X_train_rf = X_selected_train
    print("Menggunakan X_selected_test & X_selected_train untuk Random Forest (fitur terseleksi).")
elif 'X_test_pca' in globals() and 'X_train_final' in globals():
    X_test_rf = X_test_pca
    X_train_rf = X_train_final
    print("Menggunakan X_test_pca & X_train_final untuk Random Forest.")
elif 'X_test_final' in globals() and 'X_train_final' in globals():
    X_test_rf = X_test_final
    X_train_rf = X_train_final
    print("Menggunakan X_test_final & X_train_final untuk Random Forest.")
else:
    X_test_rf = X_test.copy()
    X_train_rf = X_train.copy()
    print("Menggunakan X_test & X_train asli untuk Random Forest.")

# Pilih dataset untuk XGBoost
if 'X_test_xgb_opt' in globals() and 'X_train_xgb_opt' in globals():
    X_test_xgb = X_test_xgb_opt
    X_train_xgb = X_train_xgb_opt
    print("Menggunakan X_test_xgb_opt & X_train_xgb_opt untuk XGBoost.")
else:
    X_test_xgb = X_test.copy()
    X_train_xgb = X_train.copy()
    print("Menggunakan X_test & X_train asli untuk XGBoost.")

if 'best_rf' in globals():
    rf_model = best_rf
    print("Menggunakan model 'best_rf' yang aktif di memori.")
else:
    rf_files = ['random_forest_diabetes_best.pkl', 'random_forest_diabetes_tuned.pkl', 'random_forest_diabetes.pkl']
    rf_model = None
    for file in rf_files:
        if os.path.exists(file):
            rf_model = joblib.load(file)
            print(f"Model Random Forest di-load dari file: '{file}'")
            break
    if rf_model is None:
        raise FileNotFoundError("Model Random Forest tidak ditemukan di memori RAM atau file lokal")

# Load atau ambil model XGBoost
if 'best_xgb_opt' in globals():
    xgb_model = best_xgb_opt
    print("Menggunakan model 'best_xgb_opt' dari hasil RandomizedSearchCV.")
elif 'xgb_random_search_opt' in globals() and hasattr(xgb_random_search_opt, 'best_estimator_'):
    xgb_model = xgb_random_search_opt.best_estimator_
    print("Menggunakan model best_estimator_ dari xgb_random_search_opt.")
else:
    xgb_files = ['xgboost_diabetes_best.json', 'xgboost_diabetes_tuned.json', 'xgboost_diabetes.json']
    xgb_model = None
    for file in xgb_files:
        if os.path.exists(file):
            xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1, verbosity=0)
            xgb_model.load_model(file)
            print(f"Model XGBoost di-load dari file: '{file}'")
            break
    if xgb_model is None:
        raise FileNotFoundError("Model XGBoost tidak ditemukan di memori RAM atau file lokal (.json). Harap jalankan kembali sel pelatihan XGBoost.")

print("\nMenghitung metrik klasifikasi untuk kedua model")

y_pred_rf = rf_model.predict(X_test_rf)
y_prob_rf = rf_model.predict_proba(X_test_rf)[:, 1]
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
roc_rf = roc_auc_score(y_test, y_prob_rf)

# Prediksi XGBoost
y_pred_xgb = xgb_model.predict(X_test_xgb)
y_prob_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]
acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
roc_xgb = roc_auc_score(y_test, y_prob_xgb)

print("Menghitung skor Cross-Validation")
if 'y_train_final' in globals() and X_train_rf.shape[0] == y_train_final.shape[0]:
    y_train_for_rf = y_train_final
else:
    y_train_for_rf = y_train

cv_rf = cross_val_score(rf_model, X_train_rf, y_train_for_rf, cv=3, scoring='accuracy', n_jobs=-1)
cv_xgb = cross_val_score(xgb_model, X_train_xgb, y_train, cv=3, scoring='accuracy', n_jobs=-1)

comparison_data = {
    'Metrik Evaluasi': [
        'Test Accuracy (Akurasi Uji)',
        'F1-Score (Test)',
        'Precision (Presisi)',
        'Recall (Sensitivitas)',
        'ROC AUC Score',
        'CV Accuracy (Mean)',
        'CV Accuracy (Std)'
    ],
    'Random Forest (Tuned)': [
        f"{acc_rf:.4f}",
        f"{f1_rf:.4f}",
        f"{prec_rf:.4f}",
        f"{rec_rf:.4f}",
        f"{roc_rf:.4f}",
        f"{cv_rf.mean():.4f}",
        f"{cv_rf.std():.4f}"
    ],
    'XGBoost (Tuned)': [
        f"{acc_xgb:.4f}",
        f"{f1_xgb:.4f}",
        f"{prec_xgb:.4f}",
        f"{rec_xgb:.4f}",
        f"{roc_xgb:.4f}",
        f"{cv_xgb.mean():.4f}",
        f"{cv_xgb.std():.4f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + "="*70)
print("TABEL RINGKASAN METRIK EVALUASI")
print("="*70)
print(comparison_df.to_string(index=False))

print("\n" + "-"*70)
print("CLASSIFICATION REPORT - RANDOM FOREST (TUNED)")
print("-"*70)
print(classification_report(y_test, y_pred_rf))

print("\n" + "-"*70)
print("CLASSIFICATION REPORT - XGBOOST (TUNED)")
print("-"*70)
print(classification_report(y_test, y_pred_xgb))

# Visualisasi Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Sehat (0)', 'Diabetes (1)'])
disp_rf.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Confusion Matrix: Random Forest (Tuned)', fontsize=12, fontweight='bold')

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['Sehat (0)', 'Diabetes (1)'])
disp_xgb.plot(ax=axes[1], cmap='Oranges', values_format='d')
axes[1].set_title('Confusion Matrix: XGBoost (Tuned)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

plt.figure(figsize=(10, 7))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_rf:.4f})', color='royalblue', lw=2)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_xgb:.4f})', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess (AUC = 0.5000)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=11)
plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=11)
plt.title('Kurva ROC (Receiver Operating Characteristic) - Perbandingan Model', fontsize=13, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.show()

print("ANALISIS & REKOMENDASI")

diff_acc = abs(acc_xgb - acc_rf) * 100
diff_f1 = abs(f1_xgb - f1_rf) * 100

best_model = "XGBoost" if f1_xgb > f1_rf else "Random Forest"
worse_model = "Random Forest" if best_model == "XGBoost" else "XGBoost"

print(f"Akurasi Uji: XGBoost ({acc_xgb:.4f}) vs Random Forest ({acc_rf:.4f})")
print(f"F1-Score   : XGBoost ({f1_xgb:.4f}) vs Random Forest ({f1_rf:.4f})")
print(f"ROC AUC Score: XGBoost ({roc_xgb:.4f}) vs Random Forest ({roc_rf:.4f})")

print(f"\nKESIMPULAN:")
if best_model == "XGBoost":
    print(f"XGBoost (Tuned) merupakan model terbaik karena memiliki F1-score {diff_f1:.2f}% lebih tinggi dari Random Forest.")
    print("XGBoost menunjukkan recall yang sangat baik, yang krusial untuk deteksi dini penyakit diabetes agar meminimalkan false negatives.")
else:
    print(f"Random Forest (Tuned) merupakan model terbaik karena memiliki F1-score {diff_f1:.2f}% lebih tinggi dari XGBoost.")
    print("Random Forest menunjukkan performa seimbang antara kelas Sehat dan Diabetes pada dataset ini.")

try:
    if best_model == "XGBoost":
        xgb_model.save_model('xgboost_diabetes_tuned.json')
        print("\nModel terbaik (XGBoost) disimpan sebagai 'xgboost_diabetes_tuned.json'")
    else:
        joblib.dump(rf_model, 'random_forest_diabetes_tuned.pkl')
        print("\nModel terbaik (Random Forest) disimpan sebagai 'random_forest_diabetes_tuned.pkl'")
except Exception as e:
    print(f"\nGagal menyimpan model terbaik: {e}")

print("UJI MODEL RANDOM FOREST")
if 'X_test_pca' not in globals() or 'y_test' not in globals():
    print("Mencoba memuat data uji dari file lokal...")
    if os.path.exists('X_test_pca.pkl') and os.path.exists('y_test.pkl'):
        X_test_pca = joblib.load('X_test_pca.pkl')
        y_test     = joblib.load('y_test.pkl')
        if os.path.exists('X_test.pkl'):
            X_test = joblib.load('X_test.pkl')
        print("SUCCESS: Data uji berhasil dimuat dari file disk!")
    else:
        raise NameError("Variabel X_test_pca atau y_test tidak ditemukan di memori dan file .pkl tidak lengkap. Jalankan ulang sel preprocessing dan pembagian data terlebih dahulu.")
# 2. MEMUAT MODEL RANDOM FOREST
if 'best_rf' in globals():
    rf_model = best_rf
    print("Menggunakan model 'best_rf' dari memori.")
elif os.path.exists('random_forest_diabetes_tuned.pkl'):
    rf_model = joblib.load('random_forest_diabetes_tuned.pkl')
    print("Model Random Forest di-load dari file 'random_forest_diabetes_tuned.pkl'.")
elif os.path.exists('random_forest_diabetes_best.pkl'):
    rf_model = joblib.load('random_forest_diabetes_best.pkl')
    print("Model Random Forest di-load dari file 'random_forest_diabetes_best.pkl'.")
else:
    raise FileNotFoundError("Model Random Forest tidak ditemukan di memori atau file lokal.")
# EVALUASI MODEL PADA DATA UJI
rf_y_pred = rf_model.predict(X_test_pca)
rf_y_prob = rf_model.predict_proba(X_test_pca)[:, 1]
print("\nHasil Evaluasi Random Forest pada Data Uji:")
print(f"Jumlah sampel uji: {len(y_test)}")
print(f"Akurasi: {accuracy_score(y_test, rf_y_pred):.4f}")
print(f"Precision: {precision_score(y_test, rf_y_pred):.4f}")
print(f"Recall: {recall_score(y_test, rf_y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, rf_y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, rf_y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_y_pred))
# Prediksi contoh pasien di data uji
sample_idx = 0
if isinstance(X_test_pca, np.ndarray):
    sample_X = X_test_pca[sample_idx].reshape(1, -1)
else:
    sample_X = np.array(X_test_pca.loc[sample_idx]).reshape(1, -1)
sample_pred = rf_model.predict(sample_X)[0]
sample_prob = rf_model.predict_proba(sample_X)[0, 1]
print("\nContoh Prediksi Random Forest pada satu sampel uji:")
print(f"Indeks sampel uji: {sample_idx}")
print(f"Prediksi kelas: {sample_pred}")
print(f"Probabilitas kelas 1 (diabetes): {sample_prob:.4f}")
if 'X_test' in globals():
    print("\nFitur input contoh (encoded):")
    display(X_test.iloc[[sample_idx]])

# Memuat model
if 'rf_model' not in globals():
    if 'best_rf' in globals():
        rf_model = best_rf
        print("Menggunakan best_rf dari memori.")
    elif os.path.exists('random_forest_diabetes_tuned.pkl'):
        rf_model = joblib.load('random_forest_diabetes_tuned.pkl')
        print("Memuat model dari 'random_forest_diabetes_tuned.pkl'.")
    elif os.path.exists('random_forest_diabetes_best.pkl'):
        rf_model = joblib.load('random_forest_diabetes_best.pkl')
        print("Memuat model dari 'random_forest_diabetes_best.pkl'.")
    else:
        raise FileNotFoundError('Model Random Forest tidak ditemukan di memori atau file lokal.')
if 'scaler' not in globals() or 'pca' not in globals():
    scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')
    pca_path = os.path.join(os.getcwd(), 'pca.pkl')
    if os.path.exists(scaler_path) and os.path.exists(pca_path):
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        print('Memuat scaler dan PCA dari disk untuk prediksi model tersimpan.')
    else:
        raise NameError('scaler atau pca tidak ditemukan. Pastikan model dan preprocessor telah disimpan dengan benar.')
if 'label_encoders' not in globals():
    if os.path.exists('label_encoders.pkl'):
        label_encoders = joblib.load('label_encoders.pkl')
        print("Memuat label encoders dari disk (label_encoders.pkl).")
    elif 'df' in globals():
        label_encoders = {}
        for col in ['gender', 'smoking_history']:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            label_encoders[col] = le
        print("Membuat label encoders baru dari data df di memori.")
    else:
        print("WARNING: df tidak ditemukan dan label_encoders.pkl tidak ada. Menggunakan encoder default.")
        label_encoders = {}
        
        le_gender = LabelEncoder()
        le_gender.fit(['Female', 'Male', 'Other'])
        label_encoders['gender'] = le_gender
        
        le_smoke = LabelEncoder()
        le_smoke.fit(['never', 'current', 'former', 'not current', 'ever', 'No Info'])
        label_encoders['smoking_history'] = le_smoke
feature_cols = [
    'year', 'gender', 'age',
    'race:AfricanAmerican', 'race:Asian', 'race:Caucasian',
    'race:Hispanic', 'race:Other',
    'hypertension', 'heart_disease', 'smoking_history',
    'bmi', 'hbA1c_level', 'blood_glucose_level'
]
def input_with_options(prompt, options, transform=str.strip):
    normalized = {str(opt).strip().lower(): opt for opt in options}
    while True:
        value = transform(input(prompt))
        if str(value).strip().lower() in normalized:
            return normalized[str(value).strip().lower()]
        print(' Pilihan tidak dikenal. Silakan coba lagi.')
def input_int(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt))
            if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                raise ValueError
            return value
        except ValueError:
            print(' Masukkan angka yang valid.')
def input_float(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = float(input(prompt))
            if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                raise ValueError
            return value
        except ValueError:
            print(' Masukkan angka decimal yang valid.')
print('\nTerima kasih! Silakan isi data pasien berikut:')
print('-' * 70)
print('Catatan: semua input akan diproses sesuai format yang diperlukan.')
# Input Year
year = input_int('1) Tahun (contoh: 2024): ', min_value=1900, max_value=2100)
# Input Gender
gender = input_with_options(
    '2) Gender (Male/Female): ',
    label_encoders['gender'].classes_
)
# Input Age
age = input_int('3) Usia (tahun): ', min_value=0, max_value=120)
# Input Race
grace_choices = {
    '1': 'race:AfricanAmerican',
    '2': 'race:Asian',
    '3': 'race:Caucasian',
    '4': 'race:Hispanic',
    '5': 'race:Other'
}
print('\n5) Pilih ras/etnis pasien:')
print('   1. AfricanAmerican')
print('   2. Asian')
print('   3. Caucasian')
print('   4. Hispanic')
print('   5. Other')
race_choice = input_with_options('   Masukkan nomor (1-5): ', grace_choices.keys())
race_col = grace_choices[race_choice]
race_dict_input = {
    'race:AfricanAmerican': 1 if race_col == 'race:AfricanAmerican' else 0,
    'race:Asian': 1 if race_col == 'race:Asian' else 0,
    'race:Caucasian': 1 if race_col == 'race:Caucasian' else 0,
    'race:Hispanic': 1 if race_col == 'race:Hispanic' else 0,
    'race:Other': 1 if race_col == 'race:Other' else 0,
}
# Input Hypertension
hypertension = input_with_options(
    '6) Hipertensi? (0=Tidak, 1=Ya): ',
    ['0', '1']
)
hypertension = int(hypertension)
# Input Heart Disease
heart_disease = input_with_options(
    '7) Penyakit jantung? (0=Tidak, 1=Ya): ',
    ['0', '1']
)
heart_disease = int(heart_disease)
# Input Smoking History
smoking_history = input_with_options(
    '8) Riwayat merokok (current/ever/former/never/not current): ',
    [str(x).lower() for x in label_encoders['smoking_history'].classes_],
    transform=lambda x: x.strip().lower()
)
# Input BMI
bmi = input_float('9) BMI (contoh: 28.7): ', min_value=5, max_value=70)
# Input HbA1c Level
hba1c = input_float('10) HbA1c level (contoh: 6.4): ', min_value=3.0, max_value=15.0)
# Input Blood Glucose Level
glucose = input_float('11) Blood glucose level (contoh: 140): ', min_value=40, max_value=400)
print('\n' + "="*70)
print('Memproses data dan menjalankan prediksi...')
print("="*70)
custom_input = pd.DataFrame([{
    'year': year,
    'gender': gender,
    'age': age,
    'race:AfricanAmerican': race_dict_input['race:AfricanAmerican'],
    'race:Asian': race_dict_input['race:Asian'],
    'race:Caucasian': race_dict_input['race:Caucasian'],
    'race:Hispanic': race_dict_input['race:Hispanic'],
    'race:Other': race_dict_input['race:Other'],
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'smoking_history': smoking_history,
    'bmi': bmi,
    'hbA1c_level': hba1c,
    'blood_glucose_level': glucose
}])
print('\nData input pasien:')
print(custom_input.to_string(index=False))
# Encoding
custom_encoded = custom_input.copy()
categorical_cols = ['gender', 'smoking_history']
for col in categorical_cols:
    custom_encoded[col] = label_encoders[col].transform(custom_encoded[col].astype(str))
# Scaling + PCA
custom_scaled = scaler.transform(custom_encoded[feature_cols])
custom_pca = pca.transform(custom_scaled)
# Prediksi
custom_pred = rf_model.predict(custom_pca)[0]
custom_prob = rf_model.predict_proba(custom_pca)[0, 1]
print('\n' + "="*70)
print('HASIL PREDIKSI MODEL')
print("="*70)
print(f'Prediksi kelas: {custom_pred}')
print(f'Probabilitas Diabetes (kelas 1): {custom_prob:.4f} ({custom_prob*100:.2f}%)')
print(f'Kepercayaan model: {max(custom_prob, 1-custom_prob)*100:.2f}%')
print('Interpretasi:', 'DIPREDIKSI DIABETES (1)' if custom_pred == 1 else 'DIPREDIKSI SEHAT (0)')


