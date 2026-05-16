import json
import os

notebook_path = r'e:\kuliah\Proyek ML\MachileLearning\diabetes_classification.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the new cell content
new_cell_source = [
    "# Demonstrasi Penggunaan Model (Inference)\n",
    "import joblib\n",
    "import pandas as pd\n",
    "\n",
    "# 1. Load model yang sudah disimpan\n",
    "try:\n",
    "    model = joblib.load('random_forest_diabetes_best.pkl')\n",
    "    \n",
    "    # 2. Ambil beberapa sampel dari data test untuk simulasi\n",
    "    # Menggunakan X_test_final dan y_test yang sudah ada di memori\n",
    "    samples = X_test_final.head(10)\n",
    "    real_labels = y_test.head(10).values\n",
    "\n",
    "    # 3. Lakukan prediksi\n",
    "    preds = model.predict(samples)\n",
    "    probs = model.predict_proba(samples)\n",
    "\n",
    "    # 4. Tampilkan Hasil dalam Tabel\n",
    "    inference_res = pd.DataFrame({\n",
    "        'Actual': real_labels,\n",
    "        'Predicted': preds,\n",
    "        'Probability (Diabetes)': [f\"{p[1]:.2%}\" for p in probs],\n",
    "        'Hasil': ['Benar' if p == r else 'Salah' for p, r in zip(preds, real_labels)]\n",
    "    })\n",
    "\n",
    "    print(\"=== DEMONSTRASI PREDIKSI MODEL PADA 10 DATA TEST ===\")\n",
    "    display(inference_res)\n",
    "    \n",
    "except FileNotFoundError:\n",
    "    print(\"File 'random_forest_diabetes_best.pkl' tidak ditemukan. Pastikan cell sebelumnya sudah dijalankan.\")\n",
    "except NameError as e:\n",
    "    print(f\"Error: {e}. Pastikan Anda sudah menjalankan semua cell dari atas.\")"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "inference_cell_new",
    "metadata": {},
    "outputs": [],
    "source": new_cell_source
}

# Append the new cell to the notebook
nb['cells'].append(new_cell)

# Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Cell baru untuk demo prediksi berhasil ditambahkan di paling bawah notebook!")
