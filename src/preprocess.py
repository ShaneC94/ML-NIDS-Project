import pandas as pd
import numpy as np


# FIT PREPROCESSOR (TRAINING ONLY)
def fit_preprocessor(file_paths, chunk_size=500_000):
    """
    Fit preprocessing statistics on TRAINING DATA ONLY.
    Returns:
        - feature_columns
        - drop_columns
        - global_medians
    """

    print("\n=== Fitting Preprocessor on Training Data ===")

    X_parts = []
    y_parts = []

    for file_path in file_paths:
        reader = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)

        for chunk in reader:

            chunk.columns = (
                chunk.columns
                .str.strip()
                .str.replace(" ", "_")
                .str.replace("/", "_")
                .str.lower()
            )

            label_col = [c for c in chunk.columns if c.startswith("label")][0]

            y_chunk = (
                chunk[label_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .apply(lambda x: 0 if x == "benign" else 1)
            )

            chunk.drop(columns=[label_col], inplace=True)

            # Drop leakage columns
            chunk.drop(columns=["timestamp", "dst_port"],
                       errors="ignore", inplace=True)

            chunk = chunk.apply(pd.to_numeric, errors="coerce")
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

            X_parts.append(chunk)
            y_parts.append(y_chunk)

    X_full = pd.concat(X_parts, ignore_index=True)
    y_full = pd.concat(y_parts, ignore_index=True)

    # Determine columns to drop based ONLY on training
    nan_ratio = X_full.isna().mean()
    drop_columns = nan_ratio[nan_ratio > 0.5].index.tolist()

    X_full.drop(columns=drop_columns, inplace=True)

    global_medians = X_full.median(numeric_only=True)

    X_full.fillna(global_medians, inplace=True)

    feature_columns = X_full.columns.tolist()

    print("Preprocessor fitted.")
    print("Final feature count:", len(feature_columns))

    return {
        "feature_columns": feature_columns,
        "drop_columns": drop_columns,
        "medians": global_medians.to_dict()
    }


# TRANSFORM DATA USING FROZEN PREPROCESSOR
def transform_with_preprocessor(file_path, preprocessor, chunk_size=500_000):

    feature_columns = preprocessor["feature_columns"]
    drop_columns = preprocessor["drop_columns"]
    medians = preprocessor["medians"]

    X_parts = []
    y_parts = []

    reader = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)

    for chunk in reader:

        chunk.columns = (
            chunk.columns
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("/", "_")
            .str.lower()
        )

        label_col = [c for c in chunk.columns if c.startswith("label")][0]

        y_chunk = (
            chunk[label_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .apply(lambda x: 0 if x == "benign" else 1)
        )

        chunk.drop(columns=[label_col], inplace=True)

        chunk.drop(columns=["timestamp", "dst_port"],
                   errors="ignore", inplace=True)

        chunk = chunk.apply(pd.to_numeric, errors="coerce")
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop columns decided during training
        chunk.drop(columns=[c for c in drop_columns if c in chunk.columns],
                   errors="ignore", inplace=True)

        # Add missing training columns as 0
        for col in feature_columns:
            if col not in chunk.columns:
                chunk[col] = 0

        chunk = chunk[feature_columns]

        chunk.fillna(pd.Series(medians), inplace=True)

        X_parts.append(chunk)
        y_parts.append(y_chunk)

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)

    return X, y

def load_and_preprocess_2018(file_path):
    """
    Simple preprocessing used for EDA and quick inspection.

    This does NOT use the frozen training preprocessor.
    """

    import pandas as pd
    import numpy as np

    df = pd.read_csv(file_path, low_memory=False)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.lower()
    )

    # Detect label column
    label_col = [c for c in df.columns if c.startswith("label")][0]

    y = (
        df[label_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(lambda x: 0 if x == "benign" else 1)
    )

    df.drop(columns=[label_col], inplace=True)

    # Remove leakage columns
    df.drop(columns=["timestamp", "dst_port"], errors="ignore", inplace=True)

    df = df.apply(pd.to_numeric, errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.fillna(df.median(numeric_only=True), inplace=True)

    return df, y