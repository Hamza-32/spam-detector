import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, model_name: str):
    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def full_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=["Ham", "Spam"], zero_division=0)


def make_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def results_table(rows):
    return pd.DataFrame(rows).sort_values(by=["F1-Score", "Accuracy"], ascending=False).reset_index(drop=True)


def compute_roc(y_true, y_score):
    if y_score is None:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "auc": auc}
