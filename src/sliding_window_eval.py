import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

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

# Sliding Window Temporal Experiment
def sliding_window_experiment(all_files, window_size=3, target_fpr=0.01):

    print("\n=== Sliding Window Temporal Experiment ===")

    results = []
    all_files = sorted(all_files)

    # Fit preprocessing on full training set
    preprocessor = fit_preprocessor(all_files)

    for i in range(window_size, len(all_files)):

        train_files = all_files[i-window_size:i]
        test_file = all_files[i]

        train_start = train_files[0].split("/")[-1]
        train_end = train_files[-1].split("/")[-1]
        test_day = test_file.split("/")[-1]

        print("\n------------------------------------------------")
        print(f"Sliding Window Size: {window_size}")
        print(f"Train window: {train_start} -> {train_end}")
        print(f"Testing on: {test_day}")
        print("------------------------------------------------")

        X_train_parts = []
        y_train_parts = []

        for file in train_files:

            X_part, y_part = transform_with_preprocessor(file, preprocessor)

            # convert immediately to efficient format
            X_part = X_part.astype(np.float32).to_numpy()
            y_part = y_part.to_numpy()

            X_train_parts.append(X_part)
            y_train_parts.append(y_part)

        # Efficient concatenation
        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)

        print("Training shape:", X_train.shape)
        print("Attack rate:", round(np.mean(y_train), 6))

        # Train models
        models = train_models(X_train, y_train)

        # Transform test data
        X_test, y_test = transform_with_preprocessor(test_file, preprocessor)

        X_test = X_test.astype(np.float32).to_numpy()
        y_test = y_test.to_numpy()

        # Evaluate models
        for model_name, model in models.items():

            y_scores = model.predict_proba(X_test)[:, 1]

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

            print(
                f"{model_name} | "
                f"Recall@1%FPR: {recall:.4f} | "
                f"pAUC@1%FPR: {pauc:.4f}"
            )

            results.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_day": test_day,
                "model": model_name,
                "roc_auc": auc,
                "recall@1%fpr": recall,
                "pauc@1%fpr": pauc,
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/sliding_window_results.csv",
        index=False
    )

    print("\nResults saved to results/sliding_window_results.csv")

    plot_sliding_results(results_df)

    return results_df

# Plot Sliding Window Results
def plot_sliding_results(results_df):

    plt.figure(figsize=(10, 6))

    for model in results_df["model"].unique():

        baseline_pauc = load_baseline_pauc(model)

        subset = results_df[results_df["model"] == model].sort_values("test_day")

        plt.plot(
            subset["test_day"],
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

    plt.xlabel("Test Day")
    plt.ylabel("pAUC @ 1% FPR")
    plt.title("Sliding Window Temporal Performance")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/sliding_window_pauc.png",
        dpi=300
    )
    plt.close()

    print("Saved plot to results/sliding_window_pauc.png")