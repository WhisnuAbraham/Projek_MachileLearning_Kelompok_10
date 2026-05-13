import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import json

# 1. Load Data
df = pd.read_csv('diabetes_dataset.csv')

# 2. Preprocessing
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
X_train, X_test, y_train, y_test = train_test_split(df_pca, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Feature Importance to get selected_features
rf_temp = RandomForestClassifier(random_state=42)
rf_temp.fit(X_train, y_train)
importances = rf_temp.feature_importances_
feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
selected_features = feat_imp.head(10).index.tolist()

X_train_final = X_train[selected_features]
X_test_final = X_test[selected_features]

print("="*60)
print("1. MODEL SEBELUM TUNING (BASELINE)")
print("="*60)
rf_baseline = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_final, y_train)
y_pred_baseline = rf_baseline.predict(X_test_final)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
print(f"Test Accuracy (Baseline): {baseline_accuracy:.4f}\n")

print("="*60)
print("2. METODE TUNING (GridSearchCV)")
print("="*60)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
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
grid_search.fit(X_train_final, y_train)
print(f"Best Parameters: {grid_search.best_params_}")

print("\n" + "="*60)
print("3. MODEL SETELAH TUNING (TUNED PERFORMANCE)")
print("="*60)
best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test_final)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
print(f"Test Accuracy (Tuned): {tuned_accuracy:.4f}")
print("="*60)

# Write results to a markdown artifact
with open('tuning_results.md', 'w') as f:
    f.write(f"# Hasil Hyperparameter Tuning\n\n")
    f.write(f"**1. Model Sebelum Tuning (Baseline)**\n")
    f.write(f"- Akurasi: {baseline_accuracy:.4f}\n\n")
    f.write(f"**2. Metode Tuning**\n")
    f.write(f"- Metode: GridSearchCV\n")
    f.write(f"- Parameter Terbaik: {grid_search.best_params_}\n\n")
    f.write(f"**3. Model Setelah Tuning**\n")
    f.write(f"- Akurasi: {tuned_accuracy:.4f}\n")

# Create a fixed notebook
with open('diabetes_classification.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Find the tuning cell and replace its content
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "GridSearchCV" in source and "param_grid" in source:
            cell['source'] = [
                "from sklearn.model_selection import GridSearchCV\n",
                "\n",
                "# Pastikan X_selected_train terdefinisi untuk menghindari NameError\n",
                "X_selected_train = X_train[selected_features]\n",
                "X_selected_test = X_test[selected_features]\n",
                "\n",
                "param_grid = {\n",
                "    'n_estimators': [100, 200],\n",
                "    'max_depth': [10, 20, None],\n",
                "    'min_samples_split': [2, 5]\n",
                "}\n",
                "\n",
                "rf_grid = RandomForestClassifier(random_state=42, class_weight='balanced')\n",
                "grid_search = GridSearchCV(estimator=rf_grid, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)\n",
                "\n",
                "print(\"Fitting GridSearchCV...\")\n",
                "grid_search.fit(X_selected_train, y_train)\n",
                "print(f\"Best Parameters: {grid_search.best_params_}\")\n"
            ]

with open('diabetes_classification_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
