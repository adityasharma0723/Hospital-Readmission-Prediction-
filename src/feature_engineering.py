"""
Feature Engineering: Combine NLP text features with structured clinical features.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def combine_features(X_structured, X_text):
    """Combine structured features with NLP-derived text features."""
    if hasattr(X_structured, 'values'):
        X_structured = X_structured.values
    X_combined = np.hstack([X_structured, X_text])
    return X_combined


def scale_features(X_train, X_test):
    """Scale combined features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"[INFO] Features scaled using StandardScaler")
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to handle class imbalance."""
    print(f"[INFO] Before SMOTE - Class distribution: {np.bincount(y_train)}")
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE  - Class distribution: {np.bincount(y_resampled)}")
    return X_resampled, y_resampled


def run_feature_engineering(data):
    """Execute the feature engineering pipeline."""
    print("=" * 60)
    print("  STEP 3: FEATURE ENGINEERING")
    print("=" * 60)

    print("\n--- Combining Structured + Text Features ---")
    X_train_combined = combine_features(
        data["X_train_structured"], data["X_text_train"]
    )
    X_test_combined = combine_features(
        data["X_test_structured"], data["X_text_test"]
    )

    # Build combined feature names
    n_text_features = data["X_text_train"].shape[1]
    text_feature_names = [f"text_svd_{i}" for i in range(n_text_features)]
    feature_names = data["feature_names_structured"] + text_feature_names

    print(f"[INFO] Combined features shape: {X_train_combined.shape}")
    print(f"[INFO]   Structured: {len(data['feature_names_structured'])} features")
    print(f"[INFO]   Text (SVD): {n_text_features} features")

    print("\n--- Scaling Features ---")
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_combined, X_test_combined
    )

    print("\n--- Applying SMOTE ---")
    X_train_resampled, y_train_resampled = apply_smote(
        X_train_scaled, data["y_train"]
    )

    data["X_train"] = X_train_resampled
    data["X_test"] = X_test_scaled
    data["y_train_resampled"] = y_train_resampled
    data["scaler"] = scaler
    data["feature_names"] = feature_names
    data["X_train_original"] = X_train_scaled
    data["y_train_original"] = data["y_train"]

    print("\n[Done] Feature engineering complete!\n")
    return data
