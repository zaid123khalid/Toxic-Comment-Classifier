from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def build_model(model_type="logistic", **kwargs):
    """
    Creates a binary classification model.

    Args:
        model_type: 'logistic' or 'random_forest'
        **kwargs: Additional parameters for the model
    """
    if model_type == "logistic":
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            **kwargs,
        )
    # Add other model types if needed, e.g., RandomForestClassifier with class_weight='balanced'
    # elif model_type == "random_forest":
    #     return RandomForestClassifier(
    #         n_estimators=100, random_state=42, class_weight="balanced", **kwargs
    #     )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_model(model, X_train, y_train):
    """
    Trains the model. Handles multi-label output.
    """
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Predicts toxicity labels.
    """
    return model.predict(X_test)


def predict_proba(model, X_test):
    """Returns probability of positive class (toxic)."""
    return model.predict_proba(X_test)[:, 1]


def evaluate_model(y_true, y_pred, y_proba=None):
    """
    Calculates evaluation metrics for binary classification.

    Args:
        y_true: True labels (Series or array)
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class (optional)
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except:
            metrics["roc_auc"] = None

    return metrics


def print_classification_report(y_true, y_pred, label_name="toxic"):
    """Displays classification report for binary classification."""
    print(f"\n{'='*60}")
    print(f"📊 CLASSIFICATION REPORT - {label_name.upper()}")
    print(f"{'='*60}")
    print(
        classification_report(
            y_true, y_pred, target_names=["Non-Toxic", "Toxic"], zero_division=0
        )
    )


def plot_confusion_matrix(y_true, y_pred, label_name="toxic", save_path=None):
    """Plots confusion matrix for binary classification."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Toxic", "Toxic"],
        yticklabels=["Non-Toxic", "Toxic"],
    )
    plt.title(f"Confusion Matrix - {label_name.upper()}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📷 Confusion matrix saved to {save_path}")

    plt.show()


def plot_roc_curve(y_true, y_proba, label_name="toxic", save_path=None):
    """Plots ROC curve if probabilities are available."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label_name.upper()}")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📈 ROC curve saved to {save_path}")

    plt.show()


def save_model(model, path):
    """
    Saves the trained model using joblib.
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """
    Loads the saved model.
    """
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model
