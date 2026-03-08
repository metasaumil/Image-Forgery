"""
evaluation/evaluate.py
-----------------------
Comprehensive evaluation of the forgery detection model.

Metrics:
- Accuracy, Precision, Recall, F1 score
- ROC curve and AUC
- Confusion matrix
- Per-class breakdown
- Error analysis (worst predictions)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
from tqdm import tqdm
from typing import Tuple, Optional
import os


# ─────────────────────────────────────────────
#  Collect Predictions
# ─────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model: nn.Module, loader: DataLoader,
                    device: torch.device = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Run model on the entire dataloader and collect:
    - True labels
    - Predicted labels
    - Fake probabilities (for ROC)
    - Image paths (for error analysis)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_labels    = []
    all_preds     = []
    all_probs     = []
    all_paths     = []

    for images, labels, paths in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        preds   = probs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # fake probability
        all_paths.extend(paths)

    return (np.array(all_labels), np.array(all_preds),
            np.array(all_probs), all_paths)


# ─────────────────────────────────────────────
#  Metric Report
# ─────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_probs: np.ndarray) -> dict:
    """Compute all classification metrics."""
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }

    # ROC / AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    metrics["auc"]   = auc(fpr, tpr)
    metrics["fpr"]   = fpr
    metrics["tpr"]   = tpr
    metrics["thresholds"] = thresholds

    return metrics


def print_report(y_true, y_pred, class_names=("Real", "Fake")):
    print("\n" + "="*50)
    print("  CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))


# ─────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names=("Real", "Fake"),
                          save_path: str = None, show: bool = True):
    """Plot a colored confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalize

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title, fmt in [
        (axes[0], cm,      "Confusion Matrix (counts)",   "d"),
        (axes[1], cm_norm, "Confusion Matrix (normalized)", ".2f"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float,
                   model_name: str = "Model",
                   save_path: str = None, show: bool = True):
    """Plot ROC curve."""
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, "b-", lw=2, label=f"{model_name} (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.fill_between(fpr, tpr, alpha=0.1, color="blue")
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Forgery Detection")
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
#  Error Analysis
# ─────────────────────────────────────────────

def error_analysis(y_true, y_pred, y_probs, paths, n_show: int = 8,
                   save_path: str = None):
    """
    Show the images where the model was most wrong (highest confidence mistakes).
    """
    from PIL import Image

    wrong_mask = y_true != y_pred
    wrong_indices = np.where(wrong_mask)[0]

    if len(wrong_indices) == 0:
        print("[Error Analysis] No mistakes found — perfect score!")
        return

    # Sort by confidence in the wrong answer (most confident mistakes first)
    wrong_probs = np.abs(y_probs[wrong_indices] - 0.5)  # distance from uncertain
    sorted_idx  = wrong_indices[np.argsort(wrong_probs)[::-1]]
    to_show     = sorted_idx[:n_show]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, idx in enumerate(to_show):
        if i >= len(axes):
            break
        try:
            img = Image.open(paths[idx]).convert("RGB")
            img = img.resize((224, 224))
            axes[i].imshow(np.array(img))
            true_lbl = "Real" if y_true[idx] == 0 else "Fake"
            pred_lbl = "Real" if y_pred[idx] == 0 else "Fake"
            axes[i].set_title(
                f"True: {true_lbl}\nPred: {pred_lbl} ({y_probs[idx]:.2f})",
                color="red", fontsize=9
            )
        except Exception:
            axes[i].set_title(f"[Error loading]\nidx={idx}")
        axes[i].axis("off")

    for j in range(len(to_show), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Most Confident Mistakes", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
#  Full Evaluation
# ─────────────────────────────────────────────

def full_evaluation(model: nn.Module, test_loader: DataLoader,
                    output_dir: str = "evaluation/results",
                    model_name: str = "model",
                    device: Optional[torch.device] = None):
    """
    Run complete evaluation and save all plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Evaluate] Running full evaluation on test set...")
    y_true, y_pred, y_probs, paths = get_predictions(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred, y_probs)

    print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")

    print_report(y_true, y_pred)

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred,
                          save_path=os.path.join(output_dir, f"confusion_{model_name}.png"),
                          show=True)

    # ROC curve
    plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["auc"],
                   model_name=model_name,
                   save_path=os.path.join(output_dir, f"roc_{model_name}.png"),
                   show=True)

    # Error analysis
    error_analysis(y_true, y_pred, y_probs, paths,
                   save_path=os.path.join(output_dir, f"errors_{model_name}.png"))

    return metrics
