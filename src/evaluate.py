"""
evaluate.py — Evaluation, confusion matrix, and error analysis
MNIST Digit Recognizer Project
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_device(device=None) -> torch.device:
    """Return the given device, or auto-detect (prefers GPU)."""
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_all_predictions(model, loader, device=None) -> tuple:
    """
    Run inference on a full DataLoader and collect all predictions and labels.

    Args:
        model:  Trained nn.Module in eval mode.
        loader: DataLoader (typically test_loader).
        device: torch.device.

    Returns:
        Tuple of (all_preds, all_labels, all_probs) as numpy arrays.
    """
    model.eval()
    device = resolve_device(device)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            log_probs = model(images)
            probs = torch.exp(log_probs)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_classification_report(preds: np.ndarray, labels: np.ndarray, save_path: str = None):
    """Print per-class precision, recall, F1, and support and optionally save to a file."""
    report = classification_report(labels, preds, target_names=[str(i) for i in range(10)])
    acc = (preds == labels).mean()
    
    output_str = f"\nClassification Report:\n{'=' * 60}\n{report}\nOverall accuracy: {acc*100:.4f}%\n"
    print(output_str)
    
    if save_path:
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(output_str)
        print(f"Classification report saved to {save_path}")


def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, save_path: str = None):
    """
    Plot a 10×10 confusion matrix with digit labels.

    Args:
        preds:     Predicted labels.
        labels:    True labels.
        save_path: Optional path to save figure.
    """
    cm = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=range(10), yticklabels=range(10))
    axes[0].set_title("Confusion matrix (counts)")
    axes[0].set_xlabel("Predicted digit")
    axes[0].set_ylabel("True digit")

    # Percentage
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", ax=axes[1],
                xticklabels=range(10), yticklabels=range(10))
    axes[1].set_title("Confusion matrix (% of true class)")
    axes[1].set_xlabel("Predicted digit")
    axes[1].set_ylabel("True digit")

    plt.suptitle("MNIST CNN — Confusion Matrix", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def plot_misclassified(model, loader, n: int = 25, device=None, save_path: str = None):
    """
    Display a grid of misclassified samples with true vs. predicted labels.

    Args:
        model:     Trained nn.Module.
        loader:    Test DataLoader.
        n:         Number of misclassified samples to display.
        save_path: Optional path to save figure.
    """
    model.eval()
    device = resolve_device(device)
    wrong_images, wrong_preds, wrong_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            preds = outputs.argmax(dim=1).cpu()
            mask = preds != labels
            wrong_images.extend(images[mask])
            wrong_preds.extend(preds[mask].tolist())
            wrong_labels.extend(labels[mask].tolist())
            if len(wrong_images) >= n:
                break

    n = min(n, len(wrong_images))
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    for idx, ax in enumerate(axes.flatten()):
        if idx < n:
            img = wrong_images[idx].squeeze().numpy()
            ax.imshow(img, cmap="gray")
            ax.set_title(f"True: {wrong_labels[idx]}\nPred: {wrong_preds[idx]}",
                         color="red" if wrong_labels[idx] != wrong_preds[idx] else "green",
                         fontsize=9)
        ax.axis("off")

    plt.suptitle(f"Misclassified samples (n={n})", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Misclassified grid saved to {save_path}")
    else:
        plt.show()


def full_evaluation(model, test_loader, device=None):
    """
    Run complete evaluation: accuracy, report, confusion matrix, error grid.

    Args:
        model:       Trained nn.Module.
        test_loader: Test DataLoader.
        device:      torch.device (auto-detected if None).
    """
    device = resolve_device(device)
    model = model.to(device)
    preds, labels, probs = get_all_predictions(model, test_loader, device)

    report_path = str(Path(__file__).parent.parent / "tests" / "classification_report.txt")
    print_classification_report(preds, labels, save_path=report_path)
    plot_confusion_matrix(preds, labels)
    plot_misclassified(model, test_loader, device=device)


if __name__ == "__main__":
    from dataset import get_dataloaders
    from train import load_model

    _, test_loader = get_dataloaders(batch_size=64, augment=False)
    model = load_model()
    full_evaluation(model, test_loader)
