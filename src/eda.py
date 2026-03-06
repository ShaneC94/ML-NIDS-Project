import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_eda(X, y):
    """
    Exploratory Data Analysis (EDA)

    Purpose:
    - Describe class imbalance
    - Inspect feature distributions
    - Identify missing values and anomalies
    - Produce report-ready visualizations and tables

    NOTE:
    - EDA is descriptive only
    - No transformations here affect modeling
    """

    summary = {}

    # Class distribution
    class_counts = y.value_counts()
    summary["class_counts"] = class_counts.to_dict()

    plt.figure(figsize=(6, 4))
    class_counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=["BENIGN", "MALICIOUS"], rotation=0)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/class_distribution.png", dpi=300)
    plt.close()

    # Missing value analysis
    missing_ratio = X.isna().mean().sort_values(ascending=False)
    missing_ratio = missing_ratio[missing_ratio > 0]

    summary["num_features"] = X.shape[1]
    summary["features_with_missing"] = len(missing_ratio)

    if not missing_ratio.empty:
        missing_ratio.to_csv(f"{RESULTS_DIR}/missing_value_ratio.csv")

    # Anomaly checks (basic, numeric-only)
    anomalies = {}

    if "flow_duration" in X.columns:
        anomalies["negative_flow_duration"] = int(
            (X["flow_duration"] < 0).sum()
        )

    for col in ["tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts"]:
        if col in X.columns:
            anomalies[f"{col}_zero"] = int((X[col] == 0).sum())

    summary["anomalies"] = anomalies

    # Feature distribution plots
    numeric_features = [
        "flow_duration",
        "tot_fwd_pkts",
        "tot_bwd_pkts",
        "totlen_fwd_pkts",
    ]
    numeric_features = [f for f in numeric_features if f in X.columns]

    X_eda = X[numeric_features].copy()

    # Linear scale
    X_eda.hist(bins=30, figsize=(15, 8))
    plt.suptitle("Feature Distributions (Linear Scale)")
    plt.tight_layout()
    plt.savefig(
        f"{RESULTS_DIR}/feature_distributions_linear.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Log scale (heavy-tailed behavior)
    X_eda.hist(bins=30, figsize=(15, 8), log=True)
    plt.suptitle("Feature Distributions (Log Scale)")
    plt.tight_layout()
    plt.savefig(
        f"{RESULTS_DIR}/feature_distributions_log.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save EDA summary
    summary_path = f"{RESULTS_DIR}/eda_summary.txt"
    with open(summary_path, "w") as f:
        f.write("EDA SUMMARY\n")
        f.write("=" * 40 + "\n\n")

        f.write("Class Distribution:\n")
        for k, v in summary["class_counts"].items():
            f.write(f"  Class {k}: {v}\n")

        f.write("\nMissing Values:\n")
        f.write(f"  Total features: {summary['num_features']}\n")
        f.write(
            f"  Features with missing values: "
            f"{summary['features_with_missing']}\n"
        )

        f.write("\nAnomalies:\n")
        for k, v in anomalies.items():
            f.write(f"  {k}: {v}\n")

    print(f"[EDA] Summary saved to {summary_path}")