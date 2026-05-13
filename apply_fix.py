import json

with open('diabetes_classification.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_cell = {
    "cell_type": "markdown",
    "id": "tuning_final_fixed",
    "metadata": {},
    "source": [
        "## Hyperparameter Tuning (Final & Fixed)\n",
        "Bagian ini mencakup:\n",
        "1. Model sebelum tuning\n",
        "2. Metode tuning (GridSearchCV)\n",
        "3. Model setelah tuning"
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "tuning_code_fixed",
    "metadata": {},
    "outputs": [],
    "source": [
        "from sklearn.model_selection import GridSearchCV\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
        "\n",
        "# -- 0. Persiapan Mencegah Error --\n",
        "# Pastikan fitur sudah diseleksi agar tidak muncul NameError\n",
        "X_train_final = X_train[selected_features]\n",
        "X_test_final = X_test[selected_features]\n",
        "\n",
        "# -- 1. MODEL SEBELUM TUNING (Baseline) --\n",
        "print(\"=\"*60)\n",
        "print(\"1. MODEL SEBELUM TUNING (BASELINE)\")\n",
        "print(\"=\"*60)\n",
        "rf_baseline = RandomForestClassifier(random_state=42)\n",
        "rf_baseline.fit(X_train_final, y_train)\n",
        "\n",
        "y_pred_baseline = rf_baseline.predict(X_test_final)\n",
        "baseline_accuracy = accuracy_score(y_test, y_pred_baseline)\n",
        "print(f\"Akurasi Baseline: {baseline_accuracy:.4f}\\n\")\n",
        "\n",
        "# -- 2. METODE TUNING (GridSearchCV) --\n",
        "print(\"=\"*60)\n",
        "print(\"2. METODE TUNING (Menggunakan GridSearchCV)\")\n",
        "print(\"=\"*60)\n",
        "param_grid = {\n",
        "    'n_estimators': [100, 200],\n",
        "    'max_depth': [10, 20, None],\n",
        "    'min_samples_split': [2, 5]\n",
        "}\n",
        "\n",
        "# Inisiasi model untuk tuning\n",
        "rf_grid = RandomForestClassifier(random_state=42, class_weight='balanced')\n",
        "grid_search = GridSearchCV(\n",
        "    estimator=rf_grid, \n",
        "    param_grid=param_grid, \n",
        "    cv=3, \n",
        "    scoring='f1', \n",
        "    n_jobs=-1, \n",
        "    verbose=1\n",
        ")\n",
        "\n",
        "print(\"Memulai GridSearchCV... (mohon tunggu sebentar)\\n\")\n",
        "grid_search.fit(X_train_final, y_train)\n",
        "print(f\"Parameter Terbaik: {grid_search.best_params_}\\n\")\n",
        "\n",
        "# -- 3. MODEL SETELAH TUNING --\n",
        "print(\"=\"*60)\n",
        "print(\"3. MODEL SETELAH TUNING (TUNED PERFORMANCE)\")\n",
        "print(\"=\"*60)\n",
        "best_rf = grid_search.best_estimator_\n",
        "y_pred_tuned = best_rf.predict(X_test_final)\n",
        "tuned_accuracy = accuracy_score(y_test, y_pred_tuned)\n",
        "\n",
        "print(f\"Akurasi Setelah Tuning: {tuned_accuracy:.4f}\")\n",
        "print(\"\\nClassification Report (Setelah Tuning):\")\n",
        "print(classification_report(y_test, y_pred_tuned))\n",
        "\n",
        "# Ringkasan Peningkatan\n",
        "improvement = ((tuned_accuracy - baseline_accuracy) / baseline_accuracy) * 100\n",
        "print(f\"\\nRingkasan: Ada perubahan akurasi sebesar {improvement:+.2f}% setelah tuning.\")\n"
    ]
}

nb['cells'].append(markdown_cell)
nb['cells'].append(code_cell)

with open('diabetes_classification.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Berhasil menambahkan cell tuning di akhir notebook!")
