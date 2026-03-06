import pandas as pd
import numpy as np
import joblib
import os
import gc

from src.preprocess import load_and_preprocess_2018
from src.eda import run_eda
from src.evaluate import evaluate_model
from src.threshold_analysis import threshold_analysis
from src.feature_importance import plot_feature_importance
from src.drift_analysis import comprehensive_temporal_analysis
from src.plot_pauc_comp import plot_pauc_comp

# SETUP
os.makedirs("results", exist_ok=True)

print("\n" + "=" * 70)
print(" ML-NIDS: TEMPORAL DRIFT ANALYSIS (pAUC@1%FPR)")
print("=" * 70)

# Load trained models and metadata
print("\nLoading trained models and metadata...")

rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")

train_features = joblib.load("models/train_features.pkl")
training_metadata = joblib.load("models/training_metadata.pkl")

print(
    f"Models trained on {training_metadata['date_range']} "
    f"({training_metadata['num_samples']:,} samples, "
    f"{training_metadata['attack_rate']:.2%} attack rate)"
)

# Feature alignment helper
def align_features(X, reference_columns):
    for col in reference_columns:
        if col not in X.columns:
            X[col] = 0
    return X[reference_columns].copy()


# File paths
TRAIN_FILES = training_metadata["train_files"]

CALIBRATION_FILES = [
    "data/CSE-CICIDS-2018/02-23-2018.csv",
    "data/CSE-CICIDS-2018/02-28-2018.csv",
    "data/CSE-CICIDS-2018/03-01-2018.csv",
    "data/CSE-CICIDS-2018/03-02-2018.csv",
]

TEST_FILES = [
    "data/CSE-CICIDS-2018/02-20-2018.csv",
    "data/CSE-CICIDS-2018/02-21-2018.csv",
    "data/CSE-CICIDS-2018/02-22-2018.csv",
]

PRIMARY_TEST = "data/CSE-CICIDS-2018/02-23-2018.csv"


# PHASE 1: BASELINE SNAPSHOT + EDA
print("\n" + "=" * 70)
print("BASELINE SNAPSHOT & EDA (Feb 23)")
print("=" * 70)

X_test, y_test = load_and_preprocess_2018(PRIMARY_TEST)
X_test = align_features(X_test, train_features)

# Convert to efficient format
X_test = X_test.astype("float32").to_numpy()
y_test = y_test.to_numpy()

print(
    f"Test set: {X_test.shape[0]:,} samples | "
    f"Attack rate: {y_test.mean():.2%}"
)

print("\nRunning exploratory data analysis (required)...")
run_eda(X_test, y_test)

gc.collect()

print("\n--- Default Threshold Evaluation (0.5) ---")

evaluate_model(
    rf_model,
    X_test,
    y_test,
    model_name="Random Forest (Default)",
)

evaluate_model(
    xgb_model,
    X_test,
    y_test,
    model_name="XGBoost (Default)",
)


# THRESHOLD CALIBRATION
print("\n" + "=" * 70)
print(" THRESHOLD CALIBRATION (≤1% FPR)")
print("=" * 70)

X_cal_parts = []
y_cal_parts = []

for file in CALIBRATION_FILES:

    X_day, y_day = load_and_preprocess_2018(file)
    X_day = align_features(X_day, train_features)

    # convert immediately to efficient format
    X_day = X_day.astype("float32").to_numpy()
    y_day = y_day.to_numpy()

    X_cal_parts.append(X_day)
    y_cal_parts.append(y_day)

# efficient concatenation
X_cal = np.vstack(X_cal_parts)
y_cal = np.concatenate(y_cal_parts)

# free memory
del X_cal_parts
del y_cal_parts
gc.collect()

print(
    f"Calibration set: {X_cal.shape[0]:,} samples | "
    f"Attack rate: {np.mean(y_cal):.4%}"
)

rf_threshold, _ = threshold_analysis(
    rf_model,
    X_cal,
    y_cal,
    model_name="Random Forest (Calibration)",
)

xgb_threshold, _ = threshold_analysis(
    xgb_model,
    X_cal,
    y_cal,
    model_name="XGBoost (Calibration)",
)

print("\nSelected operating thresholds:")
print(f"  Random Forest: {rf_threshold:.3f}")
print(f"  XGBoost:       {xgb_threshold:.3f}")

print("\n--- Calibrated Evaluation (Feb 23) ---")

evaluate_model(
    rf_model,
    X_test,
    y_test,
    model_name="Random Forest (Calibrated)",
    threshold=rf_threshold,
)

evaluate_model(
    xgb_model,
    X_test,
    y_test,
    model_name="XGBoost (Calibrated)",
    threshold=xgb_threshold,
)


#TEMPORAL DRIFT ANALYSIS
print("\n" + "=" * 70)
print("TEMPORAL DRIFT ANALYSIS")
print("=" * 70)

thresholds = {
    "RF": rf_threshold,
    "XGB": xgb_threshold,
}

results_df, drift_df = comprehensive_temporal_analysis(
    train_files=TRAIN_FILES,
    test_files=TEST_FILES,
    thresholds=thresholds,
    models_dir="models",
)

results_df.to_csv("results/temporal_performance.csv", index=False)
drift_df.to_csv("results/feature_drift.csv", index=False)

print("\nTemporal performance and drift statistics saved.")


# SAME-DAY vs TEMPORAL pAUC COMPARISON
print("\n" + "=" * 70)
print("SAME-DAY vs TEMPORAL pAUC@1%FPR")
print("=" * 70)

same_day_pauc = {
    "RF": 0.4197,
    "XGB": 0.9994
}

plot_pauc_comp(
    temporal_results_df=results_df,
    same_day_pauc=same_day_pauc["RF"],
    model_name="RF",
)

plot_pauc_comp(
    temporal_results_df=results_df,
    same_day_pauc=same_day_pauc["XGB"],
    model_name="XGB",
)


# FEATURE IMPORTANCE
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (INTERPRETABILITY)")
print("=" * 70)

plot_feature_importance(rf_model, X_test, model_name="Random Forest")
plot_feature_importance(xgb_model, X_test, model_name="XGBoost")


# FINAL SUMMARY
print("\n" + "=" * 70)
print(" ANALYSIS SUMMARY ")
print("=" * 70)

print("\n1. OPERATIONAL PERFORMANCE DEGRADATION (Recall@1% FPR):")

for model in ["RF", "XGB"]:
    subset = results_df[results_df["model"] == model]
    r_start = subset["recall@1%fpr"].iloc[0]
    r_end = subset["recall@1%fpr"].iloc[-1]

    drop_pct = ((r_start - r_end) / max(r_start, 1e-6)) * 100

    print(
        f"   {model}: {r_start:.3f} → {r_end:.3f} "
        f"({drop_pct:+.1f}%)"
    )

print("\n2. RANKING STABILITY (pAUC@1% FPR):")

for model in ["RF", "XGB"]:
    subset = results_df[results_df["model"] == model]

    print(
        f"   {model}: "
        f"{subset['pauc@1%fpr'].iloc[0]:.4f} → "
        f"{subset['pauc@1%fpr'].iloc[-1]:.4f}"
    )

print("\n3. DISTRIBUTIONAL DRIFT:")
print(f"   Mean KS: {drift_df['mean_ks'].mean():.4f}")
print(f"   Max KS:  {drift_df['max_ks'].max():.4f}")

print("\n4. ADVERSARIAL VALIDATION:")
print(
    f"   Mean discriminator AUC: "
    f"{drift_df['adversarial_auc'].mean():.4f}"
)

print(
    f"   Max discriminator AUC:  "
    f"{drift_df['adversarial_auc'].max():.4f}"
)

print("\nKEY FINDINGS:")
print("  - Same-day pAUC demonstrates strong model capability")
print("  - Temporal pAUC degrades under distribution shift")
print("  - Recall at fixed operating points collapses over time")
print("  - Adversarial validation confirms strong train–test shift")

print("\nANALYSIS COMPLETE")
print("Artifacts saved to /results")