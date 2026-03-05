import os
import joblib
import pandas as pd

from src.preprocess import fit_preprocessor, transform_with_preprocessor
from src.train import train_models

os.makedirs("models", exist_ok=True)

TRAIN_FILES = [
    "data/CSE-CICIDS-2018/02-14-2018.csv",
    "data/CSE-CICIDS-2018/02-15-2018.csv",
    "data/CSE-CICIDS-2018/02-16-2018.csv"
]

print("\n=== Training Temporal Models (Frozen Preprocessing) ===")
# Fit Preprocessor on Training Days
preprocessor = fit_preprocessor(TRAIN_FILES)

joblib.dump(preprocessor, "models/preprocessor.pkl")

# Transform Training Data
X_parts, y_parts = [], []

for file in TRAIN_FILES:
    X_part, y_part = transform_with_preprocessor(file, preprocessor)
    X_parts.append(X_part)
    y_parts.append(y_part)

X_train = pd.concat(X_parts, ignore_index=True)
y_train = pd.concat(y_parts, ignore_index=True)

print("Training shape:", X_train.shape)
print("Attack rate:", y_train.mean())

# Train Models
models = train_models(X_train, y_train)

joblib.dump(models["Random Forest"], "models/rf_model.pkl")
joblib.dump(models["XGBoost"], "models/xgb_model.pkl")

joblib.dump(preprocessor["feature_columns"], "models/train_features.pkl")

print("Training complete.")