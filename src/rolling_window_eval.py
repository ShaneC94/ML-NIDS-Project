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


# Rolling Window Temporal Experiment
def rolling_window_experiment(all_files, target_fpr=0.01):

    print("\n=== Rolling Window Temporal Experiment ===")

    results = []

    # Ensure chronological order
    all_files = sorted(all_files)

    # Fit preprocessing on full training set
    preprocessor = fit_preprocessor(all_files)

    # Cache transformed datasets to avoid repeated preprocessing
    cached_data = {}

    X_train = None
    y_train = None

    for i in range(3, len(all_files)):

        train_files = all_files[:i]
        test_file = all_files[i]

        train_start = train_files[0].split("/")[-1]
        train_end = train_files[-1].split("/")[-1]
        test_day = test_file.split("/")[-1]

        temporal_distance = i - 1

        print("\n------------------------------------------------")
        print(f"Training on {len(train_files)} days")
        print(f"Train window: {train_start} → {train_end}")
        print(f"Testing on: {test_day}")
        print("------------------------------------------------")

        # Incremental training data growth
        new_file = train_files[-1]

        print("Adding training data:", new_file)

        if new_file not in cached_data:

            X_new, y_new = transform_with_preprocessor(new_file, preprocessor)

            # Convert to efficient format
            X_new = X_new.astype(np.float32).to_numpy()
            y_new = y_new.to_numpy()

            cached_data[new_file] = (X_new, y_new)

        X_new, y_new = cached_data[new_file]

        if X_train is None:

            X_train = X_new
            y_train = y_new

        else:

            X_train = np.vstack([X_train, X_new])
            y_train = np.concatenate([y_train, y_new])

        print("Training shape:", X_train.shape)
        print("Attack rate:", round(np.mean(y_train), 6))

        # Train models
        models = train_models(X_train, y_train)

        # Transform test day
        if test_file not in cached_data:

            X_test, y_test = transform_with_preprocessor(
                test_file,
                preprocessor
            )

            X_test = X_test.astype(np.float32).to_numpy()
            y_test = y_test.to_numpy()

            cached_data[test_file] = (X_test, y_test)

        X_test, y_test = cached_data[test_file]

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

    results_df.to_csv(
        "results/rolling_window_results.csv",
        index=False
    )

    print("\nResults saved to results/rolling_window_results.csv")

    plot_rolling_results(results_df)

    return results_df


def plot_rolling_results(results_df):

    plt.figure(figsize=(10, 6))

    for model in results_df["model"].unique():

        baseline_pauc = load_baseline_pauc(model)

        subset = results_df[
            results_df["model"] == model
        ].sort_values("temporal_distance")

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
    plt.close()

    print("Saved plot to results/rolling_window_pauc.png")