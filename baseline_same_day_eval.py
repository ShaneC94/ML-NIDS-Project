import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from src.metrics import (
    partial_auc_at_fpr,
    recall_at_fpr
)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Controlled Same-Day Baseline Evaluation
X_test = joblib.load("models/X_test_same_day.pkl")
y_test = joblib.load("models/y_test_same_day.pkl")

rf = joblib.load("models/rf_same_day.pkl")
xgb = joblib.load("models/xgb_same_day.pkl")

MODELS = {
    "Random Forest": rf,
    "XGBoost": xgb,
}

MAX_FPR = 0.01

results = []

for name, model in MODELS.items():

    print(f"\n=== {name} (Same-Day Baseline) ===")

    y_scores = model.predict_proba(X_test)[:, 1]

    # Primary operational metrics
    pauc = partial_auc_at_fpr(
        y_test,
        y_scores,
        max_fpr=MAX_FPR
    )

    recall, threshold = recall_at_fpr(
        y_test,
        y_scores,
        target_fpr=MAX_FPR
    )

    print(f"pAUC@1% FPR: {pauc:.4f}")
    print(f"Recall@1% FPR: {recall:.4f}")
    print(f"Operational threshold@1% FPR: {threshold:.6f}")

    # Diagnostic fixed-threshold report (0.5)
    print("\n[Diagnostic] Classification report (threshold = 0.5):")

    y_pred_default = (y_scores >= 0.5).astype(int)

    report = classification_report(
        y_test,
        y_pred_default,
        output_dict=True,
        zero_division=0
    )

    print(classification_report(
        y_test,
        y_pred_default,
        zero_division=0
    ))

    # Store key metrics
    results.append({
        "model": name,
        "pauc@1%fpr": pauc,
        "recall@1%fpr": recall,
        "threshold@1%fpr": threshold,
        "precision_default": report["1"]["precision"],
        "recall_default": report["1"]["recall"],
        "f1_default": report["1"]["f1-score"]
    })

# Convert results to dataframe
results_df = pd.DataFrame(results)

# Save results
output_path = "results/same_day_baseline_metrics.csv"
results_df.to_csv(output_path, index=False)

print(f"\nSaved baseline metrics to {output_path}")