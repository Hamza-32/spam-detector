from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_confusion_matrices(confusions: dict, save_path: Path):
    n = len(confusions)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.8 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, (name, cm) in enumerate(confusions.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"],
            ax=axes[idx],
        )
        axes[idx].set_title(name)
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    for idx in range(len(confusions), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(results_df, save_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(results_df))
    width = 0.35

    ax.bar(x - width / 2, results_df["Accuracy"], width=width, label="Accuracy", color="#1F77B4")
    ax.bar(x + width / 2, results_df["F1-Score"], width=width, label="F1-Score", color="#FF7F0E")

    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(roc_map: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, roc in roc_map.items():
        if roc is None:
            continue
        ax.plot(roc["fpr"], roc["tpr"], label=f"{name} (AUC={roc['auc']:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_keywords(vectorizer, model, save_path: Path, top_n: int = 15):
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "feature_log_prob_"):
        weights = model.feature_log_prob_[1]
    elif hasattr(model, "coef_"):
        weights = model.coef_[0]
    else:
        return

    idx = np.argsort(weights)[-top_n:]
    words = [feature_names[i] for i in idx]
    vals = [weights[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(words, vals, color="#D62728")
    ax.set_title("Top Spam Keywords")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(df, save_path: Path):
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(counts.index, counts.values, color=["#2CA02C", "#D62728"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(histories: dict, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for idx, (name, history) in enumerate(histories.items()):
        if not history:
            continue
        axes[idx].plot(history.get("accuracy", []), label="train_acc")
        axes[idx].plot(history.get("val_accuracy", []), label="val_acc")
        axes[idx].set_title(f"{name.upper()} Training")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Accuracy")
        axes[idx].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
