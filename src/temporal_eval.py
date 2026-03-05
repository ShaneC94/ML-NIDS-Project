import joblib
from src.preprocess import load_and_preprocess_2018
from src.evaluate import evaluate_model


def temporal_evaluation(test_day, trained_on="Feb 14–16"):
    print("\n=== Cross-Day Temporal Evaluation ===")
    print(f"Models trained on: {trained_on}, tested on: {test_day}")

    # Load models and feature schema
    rf = joblib.load("models/rf_model.pkl")
    xgb = joblib.load("models/xgb_model.pkl")
    train_features = joblib.load("models/train_features.pkl")

    # Load and align test data
    X_test, y_test = load_and_preprocess_2018(test_day)

    for col in train_features:
        if col not in X_test.columns:
            X_test[col] = 0

    X_test = X_test[train_features]

    # Evaluation
    evaluate_model(rf, X_test, y_test, "RF Temporal")
    evaluate_model(xgb, X_test, y_test, "XGB Temporal")