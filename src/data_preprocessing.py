import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    """Load the dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def handle_missing_values(df):
    """Impute missing values."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    target = "readmitted"
    text_col = "diagnosis_text"

    # Remove non-feature columns from imputation
    for col in [target, "patient_id"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [FILL] {col}: filled {df[col].isnull().sum()} missing with median = {median_val:.2f}")

    # Fill any missing text with empty string
    if text_col in df.columns and df[text_col].isnull().sum() > 0:
        df[text_col].fillna("", inplace=True)
        print(f"  [FILL] {text_col}: filled missing with empty string")

    print(f"[INFO] Missing values remaining: {df.isnull().sum().sum()}")
    return df


def encode_features(df):
    """Encode categorical features and target."""
    target_col = "readmitted"
    text_col = "diagnosis_text"
    drop_cols = [target_col, text_col, "patient_id"]

    X_structured = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]
    diagnosis_text = df[text_col] if text_col in df.columns else None

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X_structured, drop_first=True)
    X_encoded = X_encoded.astype(float)

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"[INFO] Structured features shape after encoding: {X_encoded.shape}")
    print(f"[INFO] Target mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return X_encoded, y_encoded, le, diagnosis_text


def split_data(X, y, text_series=None, test_size=0.2, random_state=42):
    """Stratified train-test split."""
    if text_series is not None:
        X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
            X, y, text_series,
            test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        text_train, text_test = None, None

    print(f"[INFO] Train set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set:  {X_test.shape[0]} samples")
    print(f"[INFO] Train class distribution: {np.bincount(y_train)}")
    print(f"[INFO] Test class distribution:  {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test, text_train, text_test


def run_preprocessing_pipeline(filepath):
    """Execute the full preprocessing pipeline."""
    print("=" * 60)
    print("  STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    print("\n--- Loading Data ---")
    df = load_data(filepath)

    print("\n--- Handling Missing Values ---")
    df = handle_missing_values(df)

    print("\n--- Encoding Features ---")
    X_structured, y, label_encoder, diagnosis_text = encode_features(df)

    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test, text_train, text_test = split_data(
        X_structured, y, diagnosis_text
    )

    feature_names_structured = X_structured.columns.tolist()

    print("\n[Done] Preprocessing complete!\n")

    return {
        "df_raw": df,
        "X_train_structured": X_train,
        "X_test_structured": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "text_train": text_train,
        "text_test": text_test,
        "label_encoder": label_encoder,
        "feature_names_structured": feature_names_structured,
    }
