"""
Model Training: Train multiple classification models.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def build_models(random_state=42):
    """Build a dictionary of classification models."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=random_state, solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=random_state, max_depth=15,
            min_samples_split=5, min_samples_leaf=2
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, random_state=random_state,
            max_depth=5, learning_rate=0.1
        ),
        "SVM": SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=random_state
        ),
    }
    return models


def train_models(models, X_train, y_train):
    """Train all models and return the trained dict."""
    trained = {}
    for name, model in models.items():
        print(f"  Training {name}...", end=" ")
        model.fit(X_train, y_train)
        trained[name] = model
        print("Done")
    return trained


def run_model_training(data):
    """Execute the model training pipeline."""
    print("=" * 60)
    print("  STEP 4: MODEL TRAINING")
    print("=" * 60)

    print("\n--- Building Models ---")
    models = build_models()

    print("\n--- Training Models ---")
    trained_models = train_models(models, data["X_train"], data["y_train_resampled"])

    print("\n--- Generating Predictions ---")
    predictions = {}
    probabilities = {}
    for name, model in trained_models.items():
        predictions[name] = model.predict(data["X_test"])
        if hasattr(model, "predict_proba"):
            probabilities[name] = model.predict_proba(data["X_test"])[:, 1]
        elif hasattr(model, "decision_function"):
            probabilities[name] = model.decision_function(data["X_test"])
        print(f"  Predictions generated for {name}")

    data["trained_models"] = trained_models
    data["predictions"] = predictions
    data["probabilities"] = probabilities

    print("\n[Done] Model training complete!\n")
    return data
