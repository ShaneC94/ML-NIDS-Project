import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_feature_importance(
    model,
    feature_names,
    model_name="Model",
    top_n=15,
    save_csv=True
):

# Guard: model must expose feature_importances_
    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            f"{model_name} does not support feature_importances_"
        )

    importances = model.feature_importances_

    # Guard: feature alignment check
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Feature mismatch: model expects {len(importances)} features, "
            f"but received {len(feature_names)} names."
        )

    # Build importance table
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    total_importance = df["importance"].sum()
    if total_importance == 0:
        raise ValueError(
            f"{model_name} produced zero total feature importance"
        )

    df["importance_norm"] = df["importance"] / total_importance

    df = (
        df.sort_values("importance_norm", ascending=False)
          .head(top_n)
          .reset_index(drop=True)
    )

    # Plot
    plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["importance_norm"])
    plt.gca().invert_yaxis()

    plt.xlabel("Normalized Importance")
    plt.title(f"Top {top_n} Feature Importances ({model_name})")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    png_path = f"{RESULTS_DIR}/{safe_name}_feature_importance.png"

    plt.savefig(png_path, dpi=300)
    plt.close()

    # CSV export (for tables / appendix)
    if save_csv:
        csv_path = f"{RESULTS_DIR}/{safe_name}_feature_importance.csv"
        df.to_csv(csv_path, index=False)

    return df