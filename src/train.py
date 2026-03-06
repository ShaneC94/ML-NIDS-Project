import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_models(X, y):
    """
    Train supervised models on provided data ONLY.
    Designed for temporal evaluation (no random splits).

    Random Forest -> CPU
    XGBoost -> GPU
    """

    print("\n=== Training Models ===")

    # Memory Optimization
    print("Converting feature matrix to float32 for memory efficiency...")
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    # Random Forest
    print("Training Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    rf_model.fit(X, y)

    # XGBoost (GPU)
    print("Training XGBoost (GPU enabled)...")

    # Compute imbalance ratio
    neg = (y == 0).sum()
    pos = (y == 1).sum()

    raw_pos_weight = neg / max(pos, 1)

    # Cap extreme weight to prevent instability
    capped_pos_weight = min(raw_pos_weight, 100)

    print(f"Raw scale_pos_weight: {raw_pos_weight:.2f}")
    print(f"Capped scale_pos_weight: {capped_pos_weight:.2f}")

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=capped_pos_weight,
        reg_lambda=1.0,
        reg_alpha=0.5,
        min_child_weight=5,
        gamma=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,

        # GPU acceleration
        tree_method="hist",
        device="cuda",
    )

    xgb_model.fit(X, y)

    print("Model training complete.")

    return {
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }