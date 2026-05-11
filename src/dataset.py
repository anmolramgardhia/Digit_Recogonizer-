"""
dataset.py — MNIST DataLoader and transforms module
MNIST Digit Recognizer Project
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).parent.parent / "data"

# MNIST channel statistics (pre-computed)
MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081


def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    Build the image transform pipeline.

    Args:
        augment: If True, apply training augmentation (rotation + affine).
                 Always False for test/validation data.

    Returns:
        torchvision Compose transform.
    """
    base = [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]

    if augment:
        aug = [
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
        return transforms.Compose(aug + base)

    return transforms.Compose(base)


def get_dataloaders(
    batch_size: int = 64,
    augment: bool = True,
    num_workers: int = 2,
) -> tuple:
    """
    Load MNIST and return train and test DataLoaders.

    Downloads automatically on first run to the data/ directory.

    Args:
        batch_size:  Samples per batch (default 64).
        augment:     Apply augmentation to training data (default True).
        num_workers: Parallel data loading workers.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    train_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True,
        transform=get_transforms(augment=augment)
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, download=True,
        transform=get_transforms(augment=False)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Train: {len(train_dataset):,} images | Test: {len(test_dataset):,} images")
    print(f"Batch size: {batch_size} | Augmentation: {augment}")
    return train_loader, test_loader


def preprocess_image_tensor(pil_image) -> torch.Tensor:
    """
    Preprocess a PIL image for inference (same pipeline as test data).

    Args:
        pil_image: PIL Image (grayscale, any size — will be resized to 28×28).

    Returns:
        Tensor of shape (1, 1, 28, 28) ready for model.forward().
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    return transform(pil_image).unsqueeze(0)  # add batch dim


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(batch_size=64)
    images, labels = next(iter(train_loader))
    print(f"Batch shape:  {images.shape}")   # (64, 1, 28, 28)
    print(f"Labels shape: {labels.shape}")   # (64,)
    print(f"Pixel range:  [{images.min():.3f}, {images.max():.3f}]")
