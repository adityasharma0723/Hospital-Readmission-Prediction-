"""
Hyperparameter Tuning: GridSearchCV on the best model.
"""
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample"]
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [1000],
        "class_weight": ["balanced"]
    },
    "Gradient Boosting": {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "min_samples_split": [2, 5],
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
        "class_weight": ["balanced"]
    }
}


def tune_model(model, model_name, X_train, y_train, cv=5, scoring="f1"):
    """Tune hyperparameters for a given model."""
    if model_name not in PARAM_GRIDS:
        print(f"[WARN] No param grid for '{model_name}'. Skipping tuning.")
        return model, {}, 0.0

    param_grid = PARAM_GRIDS[model_name]
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"  Grid size: {total_combos} combinations x {cv} folds = {total_combos * cv} fits")

    if total_combos > 200:
        print("  [INFO] Grid too large - using RandomizedSearchCV (100 iterations)")
        search = RandomizedSearchCV(
            model, param_grid, n_iter=100, cv=cv,
            scoring=scoring, n_jobs=-1, random_state=42, verbose=0
        )
    else:
        search = GridSearchCV(
            model, param_grid, cv=cv,
            scoring=scoring, n_jobs=-1, verbose=0
        )

    search.fit(X_train, y_train)

    print(f"  Best Score (CV): {search.best_score_:.4f}")
    print(f"  Best Params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


def compare_before_after(model_before, model_after, X_test, y_test, model_name, output_dir):
    """Compare metrics before and after tuning."""
    results = []
    for label, model in [("Before Tuning", model_before), ("After Tuning", model_after)]:
        y_pred = model.predict(X_test)
        results.append({
            "Stage": label,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        })

    df = pd.DataFrame(results)
    print(f"\n{'---' * 20}")
    print(f"  HYPERPARAMETER TUNING RESULTS: {model_name}")
    print(f"{'---' * 20}")
    print(df.to_string(index=False))

    improvement = results[1]["F1-Score"] - results[0]["F1-Score"]
    print(f"\n  F1-Score Improvement: {improvement:+.4f}")

    filepath = os.path.join(output_dir, "tuning_comparison.csv")
    df.to_csv(filepath, index=False)
    print(f"[SAVED] {filepath}")

    return df


def run_hyperparameter_tuning(data, output_dir="outputs"):
    """Execute the hyperparameter tuning pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  STEP 6: HYPERPARAMETER TUNING")
    print("=" * 60)

    best_model_name = data["best_model_name"]
    best_model_original = data["best_model"]

    print(f"\n  Tuning: {best_model_name}")
    print(f"  Scoring: F1-Score (optimized for class imbalance)\n")

    from sklearn.base import clone
    model_to_tune = clone(best_model_original)

    tuned_model, best_params, best_cv_score = tune_model(
        model_to_tune, best_model_name,
        data["X_train"], data["y_train_resampled"]
    )

    compare_before_after(
        best_model_original, tuned_model,
        data["X_test"], data["y_test"],
        best_model_name, output_dir
    )

    data["tuned_model"] = tuned_model
    data["tuned_params"] = best_params
    data["tuned_predictions"] = tuned_model.predict(data["X_test"])

    print("\n[Done] Hyperparameter tuning complete!\n")
    return data
