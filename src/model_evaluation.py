"""
Model Evaluation: Metrics, plots, SHAP, and comparison tables.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)


def evaluate_single_model(name, y_true, y_pred, class_names):
    """Evaluate a single model and return metrics dict."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
    }

    print(f"\n{'---' * 17}")
    print(f"  {name}")
    print(f"{'---' * 17}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return metrics


def plot_confusion_matrices(predictions, y_test, class_names, output_dir):
    """Plot confusion matrices for all models."""
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_roc_curves(probabilities, y_test, output_dir):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(8, 6))

    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {filepath}")


def create_comparison_table(all_metrics, probabilities, y_test, output_dir):
    """Create and save model comparison table."""
    df = pd.DataFrame(all_metrics)

    roc_aucs = []
    for row in all_metrics:
        name = row["Model"]
        if name in probabilities:
            roc_aucs.append(round(roc_auc_score(y_test, probabilities[name]), 4))
        else:
            roc_aucs.append("N/A")
    df["ROC-AUC"] = roc_aucs

    df = df.sort_values("F1-Score", ascending=False).reset_index(drop=True)

    print(f"\n{'=' * 70}")
    print("  MODEL COMPARISON TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    best_model = df.iloc[0]["Model"]
    best_f1 = df.iloc[0]["F1-Score"]
    print(f"\n  Best Model: {best_model} (F1-Score = {best_f1})")

    filepath = os.path.join(output_dir, "model_comparison.csv")
    df.to_csv(filepath, index=False)
    print(f"[SAVED] {filepath}")

    return df, best_model


def plot_feature_importance(model, feature_names, output_dir, top_n=20):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print("[WARN] Model does not have feature_importances_. Skipping.")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.title("Top Feature Importances (Structured + NLP)", fontsize=14, fontweight="bold")
    plt.barh(
        range(len(indices)),
        importances[indices][::-1],
        color="steelblue", edgecolor="navy"
    )
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {filepath}")

    fi_df = pd.DataFrame({
        "Feature": [feature_names[i] for i in indices],
        "Importance": importances[indices]
    })
    return fi_df


def run_model_evaluation(data, output_dir="outputs"):
    """Execute the model evaluation pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  STEP 5: MODEL EVALUATION")
    print("=" * 60)

    y_test = data["y_test"]
    predictions = data["predictions"]
    probabilities = data["probabilities"]
    class_names = list(data["label_encoder"].classes_)

    print("\n--- Per-Model Metrics ---")
    all_metrics = []
    for name, y_pred in predictions.items():
        metrics = evaluate_single_model(name, y_test, y_pred, class_names)
        all_metrics.append(metrics)

    print("\n--- Confusion Matrices ---")
    plot_confusion_matrices(predictions, y_test, class_names, output_dir)

    print("\n--- ROC-AUC Curves ---")
    plot_roc_curves(probabilities, y_test, output_dir)

    comparison_df, best_model_name = create_comparison_table(
        all_metrics, probabilities, y_test, output_dir
    )

    best_model = data["trained_models"][best_model_name]
    print("\n--- Feature Importance ---")
    fi_df = plot_feature_importance(best_model, data["feature_names"], output_dir)

    data["comparison_df"] = comparison_df
    data["best_model_name"] = best_model_name
    data["best_model"] = best_model
    data["feature_importance"] = fi_df

    print("\n[Done] Model evaluation complete!\n")
    return data
