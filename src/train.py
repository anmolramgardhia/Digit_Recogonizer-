"""
train.py — Training loop and hyperparameters
MNIST Digit Recognizer Project

Run with:
    python src/train.py                  # auto-detect GPU
    python src/train.py --device cuda    # force CUDA GPU
    python src/train.py --device cpu     # force CPU
    python src/train.py --gpu-info       # print GPU diagnostic and exit
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import DigitCNN, DigitCNNBatchNorm, count_parameters

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE    = 128     # increased from 64 — larger batches make better use of GPU parallelism
EPOCHS        = 10
LEARNING_RATE = 0.001
DROPOUT_RATE  = 0.5
AUGMENT       = True
MODEL_PATH    = Path(__file__).parent.parent / "models" / "mnist_cnn.pt"


def get_device(force: str = None) -> torch.device:
    """
    Resolve the compute device with full diagnostic output.

    Priority order:
        1. --device flag (explicit override)
        2. CUDA GPU (if available)
        3. CPU fallback

    Args:
        force: "cuda", "cpu", or None for auto-detect.

    Returns:
        torch.device ready for use.
    """
    print("\n" + "=" * 55)
    print("  GPU / Device Diagnostic")
    print("=" * 55)
    print(f"  PyTorch version      : {torch.__version__}")
    print(f"  CUDA compiled in     : {torch.version.cuda or 'No (CPU-only build)'}")
    print(f"  CUDA available       : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"  GPU count            : {n}")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024 ** 3
            print(f"  GPU {i}               : {props.name}")
            print(f"    VRAM             : {mem_gb:.1f} GB")
            print(f"    CUDA capability  : {props.major}.{props.minor}")
            print(f"    Multiprocessors  : {props.multi_processor_count}")
    else:
        print("\n  ⚠  No CUDA GPU detected.")
        print("     Most likely cause: CPU-only PyTorch installed.")
        print("     Fix — uninstall and reinstall with CUDA support:")
        print()
        print("     # For CUDA 12.1 (most common for RTX cards):")
        print("     pip uninstall torch torchvision -y")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print()
        print("     # For CUDA 11.8:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print()
        print("     Then verify with: python -c \"import torch; print(torch.cuda.is_available())\"")

    # Resolve final device
    if force == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "You passed --device cuda but no CUDA GPU is available.\n"
                "See diagnostic above for the fix."
            )
        device = torch.device("cuda")
    elif force == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  ✓  Using device: {device}")
    if device.type == "cuda":
        print(f"     GPU name: {torch.cuda.get_device_name(0)}")
        print(f"     VRAM free: {torch.cuda.mem_get_info(0)[0] / 1024**3:.1f} GB")
    print("=" * 55 + "\n")

    return device


def train_epoch(model, loader, optimizer, criterion, device) -> tuple:
    """
    Run one training epoch.

    Args:
        model:     nn.Module to train.
        loader:    Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device:    torch.device.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device) -> tuple:
    """
    Evaluate model on a DataLoader (no gradient computation).

    Args:
        model:     Trained nn.Module.
        loader:    DataLoader (val or test).
        criterion: Loss function.
        device:    torch.device.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def plot_curves(history: dict, save_path: str = None):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history:   Dict with keys: train_loss, train_acc, val_loss, val_acc.
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   marker="o")
    axes[0].set_title("Loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train", marker="o")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val",   marker="o")
    axes[1].set_title("Accuracy per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("MNIST CNN — Training curves", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()


def train(model_class=DigitCNN, save: bool = True, device: torch.device = None) -> dict:
    """
    Full training pipeline.

    Args:
        model_class: Model class to instantiate (DigitCNN or DigitCNNBatchNorm).
        save:        Whether to save the best model to disk.
        device:      torch.device to train on. Defaults to auto-detected device.

    Returns:
        History dict with per-epoch metrics.
    """
    if device is None:
        device = get_device()

    # Use more workers on GPU (disk I/O is never the bottleneck on GPU)
    num_workers = 4 if device.type == "cuda" else 2
    pin_memory  = device.type == "cuda"   # speeds up CPU→GPU transfer

    train_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, augment=AUGMENT,
        num_workers=num_workers,
    )

    model = model_class(dropout=DROPOUT_RATE).to(device)
    print(f"Model : {model_class.__name__} | Parameters: {count_parameters(model):,}")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>11} {'Val Loss':>10} {'Val Acc':>10}")
    print("─" * 55)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Show VRAM usage on GPU runs
        vram_str = ""
        if device.type == "cuda":
            used  = torch.cuda.memory_allocated(0) / 1024 ** 2
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            vram_str = f"  VRAM {used:.0f}/{total:.0f} MB"

        print(f"{epoch:>6} {tr_loss:>12.4f} {tr_acc*100:>10.2f}%"
              f" {val_loss:>10.4f} {val_acc*100:>9.2f}%{vram_str}")

        if save and val_acc > best_acc:
            best_acc = val_acc
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest val accuracy: {best_acc*100:.2f}%")
    if save:
        print(f"Model saved to {MODEL_PATH}")

    plot_curves(history)
    return history


def load_model(model_class=DigitCNN, path: str = None, device: torch.device = None) -> nn.Module:
    """
    Load a saved model from disk onto the target device.

    Args:
        model_class: Model class to instantiate.
        path:        Optional path override.
        device:      Device to load model onto. Defaults to auto-detected.

    Returns:
        Loaded nn.Module in eval mode on the target device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_path = Path(path) if path else MODEL_PATH
    if not load_path.exists():
        raise FileNotFoundError(f"No model at {load_path}. Run train.py first.")

    model = model_class(dropout=DROPOUT_RATE)
    # map_location ensures the model loads correctly regardless of which
    # device it was saved on (e.g. saved on GPU, loading on CPU or vice versa)
    model.load_state_dict(torch.load(load_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST digit classifier")
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default=None,
        help="Force a specific device. Default: auto-detect (prefers GPU)."
    )
    parser.add_argument(
        "--gpu-info", action="store_true",
        help="Print GPU diagnostic info and exit."
    )
    args = parser.parse_args()

    if args.gpu_info:
        get_device(force=args.device)
        sys.exit(0)

    device = get_device(force=args.device)
    train(model_class=DigitCNN, save=True, device=device)
