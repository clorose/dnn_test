import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from sklearn.metrics import confusion_matrix, roc_curve, auc
import logging


def plot_training_history(
    history: Dict[str, Any], output_dir: str, timestamp: str, logger: logging.Logger
) -> None:
    """학습 히스토리를 시각화하고 저장합니다."""
    plots_dir = os.path.join(output_dir, "plots")

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], linewidth=2)
    plt.plot(history.history["val_accuracy"], linewidth=2)
    plt.title("Model Accuracy", fontsize=14, pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(["Training", "Validation"], fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"accuracy_{timestamp}.png"), dpi=100)
    plt.close()
    logger.info(f"Saved accuracy plot to {plots_dir}")

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], linewidth=2)
    plt.plot(history.history["val_loss"], linewidth=2)
    plt.title("Model Loss", fontsize=14, pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(["Training", "Validation"], fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"loss_{timestamp}.png"), dpi=100)
    plt.close()
    logger.info(f"Saved loss plot to {plots_dir}")


def plot_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    timestamp: str,
    logger: logging.Logger,
) -> None:
    """평가 메트릭을 시각화합니다."""
    plots_dir = os.path.join(output_dir, "plots")

    # Confusion Matrix
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14, pad=15)

    # 축 레이블 설정
    classes = ["Negative", "Positive"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    plt.colorbar()

    # 값 표시
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            fontsize=12,
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"confusion_matrix_{timestamp}.png"), dpi=100)
    plt.close()
    logger.info(f"Saved confusion matrix to {plots_dir}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true[:, 1], y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"roc_{timestamp}.png"), dpi=100)
    plt.close()
    logger.info(f"Saved ROC curve to {plots_dir}")
