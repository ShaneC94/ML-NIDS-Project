import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.preprocess import fit_preprocessor, transform_with_preprocessor
from src.train import train_models
from src.metrics import recall_at_fpr, partial_auc_at_fpr
from sklearn.metrics import roc_auc_score

os.makedirs("results", exist_ok=True)

def load_baseline_pauc(model_name):
    baseline = pd.read_csv("results/same_day_baseline_metrics.csv")

    value = baseline.loc[
        baseline["model"] == model_name,
        "pauc@1%fpr"
    ]

    if len(value) == 0:
        raise ValueError(f"Baseline pAUC not found for {model_name}")

    return float(value.values[0])

# Rolling Window Temporal Experiment (Expanding Window)
def rolling_window_experiment(all_files, target_fpr=0.01):
    """
    Expanding window temporal evaluation.

    Train on days [0:i]
    Test on day i
    Requires minimum 3 training days.
    """

    print("\n=== Rolling Window Temporal Experiment ===")

    results = []

    # Ensure chronological order
    all_files = sorted(all_files)

    for i in range(3, len(all_files)):

        train_files = all_files[:i]
        test_file = all_files[i]

        train_start = train_files[0].split("/")[-1]
        train_end = train_files[-1].split("/")[-1]
        test_day = test_file.split("/")[-1]

        temporal_distance = i - 1  # distance from first train day

        print("\n------------------------------------------------")
        print(f"Training on {len(train_files)} days")
        print(f"Train window: {train_start} → {train_end}")
        print(f"Testing on: {test_day}")
        print("------------------------------------------------")

        # Fit Preprocessing on Current Training Window
        preprocessor = fit_preprocessor(train_files)

        # Transform Training Data
        X_train_parts = []
        y_train_parts = []

        for file in train_files:
            X_part, y_part = transform_with_preprocessor(file, preprocessor)
            X_train_parts.append(X_part)
            y_train_parts.append(y_part)

        X_train = pd.concat(X_train_parts, ignore_index=True)
        y_train = pd.concat(y_train_parts, ignore_index=True)

        print("Training shape:", X_train.shape)
        print("Attack rate:", round(y_train.mean(), 6))

        # Train Models
        models = train_models(X_train, y_train)

        # Transform Test Day
        X_test, y_test = transform_with_preprocessor(
            test_file,
            preprocessor
        )

        # Evaluate Each Model
        for model_name, model in models.items():

            y_scores = model.predict_proba(X_test)[:, 1]

            # Diagnostics
            auc = roc_auc_score(y_test, y_scores)
            print(f"{model_name} | ROC-AUC: {auc:.4f}")
            
            recall, _ = recall_at_fpr(
                y_test,
                y_scores,
                target_fpr=target_fpr
            )

            pauc = partial_auc_at_fpr(
                y_test,
                y_scores,
                max_fpr=target_fpr
            )

            results.append({
                "train_days": len(train_files),
                "train_start": train_start,
                "train_end": train_end,
                "test_day": test_day,
                "temporal_distance": temporal_distance,
                "model": model_name,
                "recall@1%fpr": recall,
                "pauc@1%fpr": pauc,
            })

            print(
                f"{model_name} | "
                f"Recall@1%FPR: {recall:.4f} | "
                f"pAUC@1%FPR: {pauc:.4f}"
            )

    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(
        "results/rolling_window_results.csv",
        index=False
    )

    print("\nResults saved to results/rolling_window_results.csv")

    # Plot Results
    plot_rolling_results(results_df)

    return results_df


# Plot Rolling Window Results
def plot_rolling_results(results_df):

    plt.figure(figsize=(10, 6))

    for model in results_df["model"].unique():

        baseline_pauc = load_baseline_pauc(model)

        subset = results_df[results_df["model"] == model].sort_values("temporal_distance")

        plt.plot(
            subset["temporal_distance"],
            subset["pauc@1%fpr"],
            marker="o",
            linewidth=2,
            label=f"{model} Temporal"
        )

        plt.axhline(
            y=baseline_pauc,
            linestyle="--",
            linewidth=2,
            alpha=0.6,
        )

    plt.xlabel("Temporal Distance (Days from Start)")
    plt.ylabel("pAUC @ 1% FPR")
    plt.title("Rolling Window Temporal Degradation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/rolling_window_pauc.png",
        dpi=300
    )

    plt.show()
    plt.close()

    print("Saved plot to results/rolling_window_pauc.png")