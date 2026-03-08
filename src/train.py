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
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    print(f"Training data shape: {X.shape}")

    # Random Forest
    print("Training Random Forest...")

    #Using subsample data for RF to reduce RAM usage
    rf_sample_size = min(len(X), 2_000_000)  # Cap at 2 million samples
    rng = np.random.default_rng(42)
    rf_indices = rng.choice(len(X), rf_sample_size, replace=False)
    
    X_rf = X[rf_indices]
    y_rf = y[rf_indices]

    print(f"RF training samples: {len(X_rf)}")

    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )

    rf_model.fit(X_rf, y_rf)

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
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=capped_pos_weight,
        reg_lambda=1.0,
        reg_alpha=0.5,
        min_child_weight=5,
        gamma=1.0,
        max_delta_step=2,
        eval_metric="logloss",
        random_state=42,

        # GPU acceleration
        tree_method="hist",
        device="cuda"
    )

    xgb_model.fit(X, y)

    print("Model training complete.")

    return {
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }