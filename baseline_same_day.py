import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.preprocess import load_and_preprocess_2018
from src.train import train_models


# Controlled Same-Day Baseline Training
DAY_FILE = "data/CSE-CICIDS-2018/02-16-2018.csv"

print("Loading same-day data...")
X, y = load_and_preprocess_2018(DAY_FILE)

print("Full day shape:", X.shape)
print("Class distribution:\n", y.value_counts(normalize=True))

# SAME-DAY 80 / 20 STRATIFIED SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Train models (NO undersampling - diagnostic upper bound)
models = train_models(
    X_train,
    y_train
)

# Save models + test set
joblib.dump(models["Random Forest"], "models/rf_same_day.pkl")
joblib.dump(models["XGBoost"], "models/xgb_same_day.pkl")

joblib.dump(X_test, "models/X_test_same_day.pkl")
joblib.dump(y_test, "models/y_test_same_day.pkl")

print("Same-day baseline training complete.")