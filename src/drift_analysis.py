import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter

from src.preprocess import transform_with_preprocessor
from src.adversarial_validation import adversarial_validation
from src.metrics import recall_at_fpr, partial_auc_at_fpr


# Comprehensive Temporal Analysis (Frozen Preprocessing)

def comprehensive_temporal_analysis(
    train_files,
    test_files,
    thresholds,
    models_dir="models",
):

    print("\n=== Loading Models and Preprocessor ===")

    rf = joblib.load(f"{models_dir}/rf_model.pkl")
    xgb = joblib.load(f"{models_dir}/xgb_model.pkl")
    preprocessor = joblib.load(f"{models_dir}/preprocessor.pkl")

    # Load training data using frozen preprocessing
    X_train_parts = []
    y_train_parts = []

    for file in train_files:
        X_part, y_part = transform_with_preprocessor(file, preprocessor)
        X_train_parts.append(X_part)
        y_train_parts.append(y_part)

    X_train = pd.concat(X_train_parts, ignore_index=True)
    y_train = pd.concat(y_train_parts, ignore_index=True)

    results = []
    drift_summary = []

    # Evaluate Each Future Day
    for test_file in test_files:

        day = test_file.split("/")[-1].replace(".csv", "")
        print(f"\n=== Evaluating Day: {day} ===")

        X_test, y_test = transform_with_preprocessor(
            test_file,
            preprocessor
        )

        # Adversarial validation
        adv_auc = adversarial_validation(X_train, X_test)

        # Model Predictions
        preds = {
            "RF": (rf.predict_proba(X_test)[:, 1], thresholds["RF"]),
            "XGB": (xgb.predict_proba(X_test)[:, 1], thresholds["XGB"]),
        }

        for model_name, (proba, thr) in preds.items():

            # Operational metrics
            recall_fpr, _ = recall_at_fpr(
                y_test,
                proba,
                target_fpr=0.01
            )

            pauc = partial_auc_at_fpr(
                y_test,
                proba,
                max_fpr=0.01
            )

            # Diagnostic fixed-threshold predictions
            y_pred = (proba >= thr).astype(int)

            results.append({
                "day": day,
                "model": model_name,
                "recall@1%fpr": recall_fpr,
                "pauc@1%fpr": pauc,
                "precision": precision_score(
                    y_test, y_pred, zero_division=0
                ),
                "recall": recall_score(
                    y_test, y_pred, zero_division=0
                ),
                "f1": f1_score(
                    y_test, y_pred, zero_division=0
                ),
            })

        # Feature Drift (KS)
        drift = measure_feature_drift(X_train, X_test)

        drift_summary.append({
            "day": day,
            "adversarial_auc": adv_auc,
            **drift,
        })

    results_df = pd.DataFrame(results)
    drift_df = pd.DataFrame(drift_summary)

    plot_temporal_degradation(results_df)
    analyze_drift_impact(results_df, drift_df)

    return results_df, drift_df


# Feature Drift Measurement
def measure_feature_drift(X_train, X_test):

    ks_stats = {}

    for col in X_train.columns:
        try:
            stat, _ = ks_2samp(X_train[col], X_test[col])
            ks_stats[col] = stat
        except Exception:
            ks_stats[col] = np.nan

    sorted_feats = sorted(
        ks_stats.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "mean_ks": np.nanmean(list(ks_stats.values())),
        "max_ks": np.nanmax(list(ks_stats.values())),
        "top_drifted_features": [
            f for f, _ in sorted_feats[:10]
        ],
    }


# Temporal Degradation Plot
def plot_temporal_degradation(results_df):

    plt.figure(figsize=(10, 6))

    for model in results_df["model"].unique():

        subset = results_df[
            results_df["model"] == model
        ]

        plt.plot(
            subset["day"],
            subset["f1"],
            marker="o",
            linewidth=2,
            label=model,
        )

    plt.xlabel("Test Day")
    plt.ylabel("F1 Score (Diagnostic)")
    plt.title("Threshold Collapse Under Temporal Drift")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/temporal_degradation_f1.png", dpi=300)
    plt.close()

# Drift Impact Analysis
def analyze_drift_impact(results_df, drift_df):

    print("\n=== Drift Impact Analysis ===")

    print("\n1. Ranking Stability (pAUC@1%FPR):")
    for model in results_df["model"].unique():
        subset = results_df[results_df["model"] == model]
        print(
            f"  {model}: "
            f"{subset['pauc@1%fpr'].iloc[0]:.3f} → "
            f"{subset['pauc@1%fpr'].iloc[-1]:.3f}"
        )

    print("\n2. Distributional Drift:")
    print(f"  Mean KS: {drift_df['mean_ks'].mean():.4f}")
    print(f"  Max KS:  {drift_df['max_ks'].max():.4f}")

    print("\n3. Adversarial Validation:")
    print(
        f"  Mean discriminator AUC: "
        f"{drift_df['adversarial_auc'].mean():.4f}"
    )
    print(
        f"  Max discriminator AUC:  "
        f"{drift_df['adversarial_auc'].max():.4f}"
    )

    print("\nMost frequently drifted features:")
    all_feats = []
    for feats in drift_df["top_drifted_features"]:
        all_feats.extend(feats)

    for feat, count in Counter(all_feats).most_common(10):
        print(f"  - {feat}: {count} days")