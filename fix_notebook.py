import json
import os

notebook_path = r'e:\kuliah\Proyek ML\MachileLearning\diabetes_classification.ipynb'

if not os.path.exists(notebook_path):
    print(f"File not found: {notebook_path}")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the consolidated imports
consolidated_imports = [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "import joblib\n",
    "from google.colab import drive\n",
    "\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder\n",
    "from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, \n",
    "                             roc_auc_score, roc_curve, recall_score, ConfusionMatrixDisplay)\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "print(\"Semua library berhasil diimport!\")"
]

# 1. Update the second code cell (index 1)
# Note: In the user's notebook, the first cell is pip install, second is imports.
code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']

if len(code_cells) > 1:
    code_cells[1]['source'] = consolidated_imports

# 2. Remove imports from other cells
import_prefixes = ["import ", "from sklearn", "from imblearn", "from google.colab"]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Skip the second code cell (which we just updated)
        if cell == code_cells[1]:
            continue
            
        new_source = []
        for line in cell['source']:
            # Keep the line if it doesn't start with an import prefix
            # and isn't just a blank line after an import
            is_import = any(line.strip().startswith(prefix) for prefix in import_prefixes)
            if not is_import:
                new_source.append(line)
        
        # Clean up leading newlines that might be left over
        while new_source and new_source[0].strip() == "":
            new_source.pop(0)
            
        cell['source'] = new_source

# Save the modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook berhasil diperbarui! Semua import telah dipindahkan ke Cell 2.")
