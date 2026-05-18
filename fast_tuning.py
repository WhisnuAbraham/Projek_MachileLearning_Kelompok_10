import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import xgboost as xgb
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

# ============ LOAD & PREPROCESS DATA ============
df = pd.read_csv('diabetes_dataset.csv')

# Handle Noise
df['smoking_history'] = df['smoking_history'].replace(
    'No Info', 
    df[df['smoking_history'] != 'No Info']['smoking_history'].mode()[0]
)

# Handle Outlier
def remove_outlier_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

for col in ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level']:
    df = remove_outlier_iqr(df, col)

# Encode
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if df[col].nunique() <= 5:
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    else:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Scaling
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'diabetes' in numeric_cols:
    numeric_cols.remove('diabetes')
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# SMOTE
X = df.drop('diabetes', axis=1)
y = df['diabetes']
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# PCA
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_res)
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df_pca, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# Feature Selection
rf_temp = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_temp.fit(X_train, y_train)
importances = rf_temp.feature_importances_
feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
selected_features = feat_imp.head(10).index.tolist()

X_train_final = X_train[selected_features]
X_test_final = X_test[selected_features]

print("="*70)
print("SOLUSI CEPAT: RandomizedSearchCV vs GridSearchCV")
print("="*70)

# ============ METHOD 1: RANDOMIZED SEARCH (CEPAT) ============
print("\n1️⃣ RANDOMIZED SEARCH CV (REKOMENDASI - CEPAT)")
print("-" * 70)

param_dist = {
    'n_estimators': [50, 100, 150, 200],  # Lebih sedikit opsi
    'max_depth': [10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# RandomizedSearchCV: cukup sample 20 kombinasi random
rf_random = RandomForestClassifier(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(
    estimator=rf_random,
    param_distributions=param_dist,
    n_iter=20,  # Hanya 20 kombinasi random (dari total 480+)
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

print("Training RandomizedSearchCV (20 iterasi, 60 model fits)...")
random_search.fit(X_train_final, y_train)
y_pred_random = random_search.predict(X_test_final)
random_acc = accuracy_score(y_test, y_pred_random)

print(f"\n✅ Best Params (RandomizedSearchCV): {random_search.best_params_}")
print(f"✅ Best CV Score: {random_search.best_score_:.4f}")
print(f"✅ Test Accuracy: {random_acc:.4f}")

# ============ METHOD 2: REDUCED GRIDSEARCH ============
print("\n2️⃣ REDUCED GRID SEARCH CV (ALTERNATIF)")
print("-" * 70)

param_grid_small = {
    'n_estimators': [100, 200],  # Hanya 2
    'max_depth': [10, 20],       # Hanya 2 (bukan 3)
    'min_samples_split': [2, 5], 
    'min_samples_leaf': [1, 2]
}

# Total: 2×2×2×2 = 16 kombinasi, 16×3 = 48 model fits (20% lebih cepat)
rf_grid = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search_small = GridSearchCV(
    estimator=rf_grid,
    param_grid=param_grid_small,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("Training GridSearchCV (16 kombinasi, 48 model fits)...")
grid_search_small.fit(X_train_final, y_train)
y_pred_grid = grid_search_small.predict(X_test_final)
grid_acc = accuracy_score(y_test, y_pred_grid)

print(f"\n✅ Best Params (GridSearchCV): {grid_search_small.best_params_}")
print(f"✅ Best CV Score: {grid_search_small.best_score_:.4f}")
print(f"✅ Test Accuracy: {grid_acc:.4f}")

# ============ METHOD 3: XGBOOST (ALTERNATIF ALGORITMA) ============
print("\n3️⃣ XGBOOST DENGAN RANDOMIZED SEARCH (PALING CEPAT)")
print("-" * 70)

param_dist_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist_xgb,
    n_iter=15,  # Hanya 15 iterasi
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

print("Training XGBoost RandomizedSearchCV (15 iterasi, 45 model fits)...")
xgb_search.fit(X_train_final, y_train)
y_pred_xgb = xgb_search.predict(X_test_final)
xgb_acc = accuracy_score(y_test, y_pred_xgb)

print(f"\n✅ Best Params (XGBoost): {xgb_search.best_params_}")
print(f"✅ Best CV Score: {xgb_search.best_score_:.4f}")
print(f"✅ Test Accuracy: {xgb_acc:.4f}")

# ============ COMPARISON ============
print("\n" + "="*70)
print("📊 PERBANDINGAN KECEPATAN & AKURASI")
print("="*70)
print(f"RandomizedSearchCV (20 iter):     Acc={random_acc:.4f}, Model fits=60")
print(f"GridSearchCV Reduced (16 params): Acc={grid_acc:.4f}, Model fits=48")
print(f"XGBoost RandomSearch (15 iter):   Acc={xgb_acc:.4f}, Model fits=45")

print("\n💡 REKOMENDASI:")
print("   - Gunakan RandomizedSearchCV untuk dataset besar")
print("   - Gunakan XGBoost untuk kecepatan maksimal")
print("   - GridSearchCV hanya untuk parameter kecil (<10 kombinasi)")
