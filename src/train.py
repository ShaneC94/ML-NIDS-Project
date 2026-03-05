from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_models(X, y):
    """
    Train supervised models on provided data ONLY.
    Designed for temporal evaluation (no random splits).
    Includes stabilized class weighting.
    """

    print("\n=== Training Models ===")
    # Random Forest
    print("Training Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    rf_model.fit(X, y)

    # XGBoost
    print("Training XGBoost...")

    # Compute imbalance ratio
    neg = (y == 0).sum()
    pos = (y == 1).sum()

    raw_pos_weight = neg / max(pos, 1)

    # Cap extreme weight to prevent gradient instability
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
        tree_method="hist"
    )

    xgb_model.fit(X, y)

    print("Model training complete.")

    return {
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }