"""
model.py — CNN architecture definition
MNIST Digit Recognizer Project

Architecture (from PRD Section 3):
  Conv1 (32 filters, 3×3, ReLU) → MaxPool → Conv2 (64 filters, 3×3, ReLU)
  → MaxPool → Flatten → Dropout(0.5) → FC(128, ReLU) → FC(10, Softmax)

Total params: ~224,834
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """
    Two-block CNN for MNIST digit classification.

    Input:  (N, 1, 28, 28) — batch of grayscale images
    Output: (N, 10)         — log-softmax class probabilities
    """

    def __init__(self, dropout: float = 0.5):
        """
        Args:
            dropout: Dropout probability (default 0.5).
        """
        super().__init__()

        # Block 1: Conv → ReLU → MaxPool
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: Conv → ReLU → MaxPool
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier head
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (N, 1, 28, 28).

        Returns:
            Log-softmax probabilities (N, 10).
        """
        # Block 1: 1×28×28 → 32×26×26 → 32×13×13
        x = self.pool1(F.relu(self.conv1(x)))

        # Block 2: 32×13×13 → 64×11×11 → 64×5×5
        x = self.pool2(F.relu(self.conv2(x)))

        # Flatten: 64×5×5 → 1600
        x = x.view(x.size(0), -1)

        # FC head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class DigitCNNBatchNorm(nn.Module):
    """
    CNN variant with Batch Normalisation after each conv layer.

    Typically converges faster and achieves slightly higher accuracy
    than the base model. Included for architecture comparison.

    Input:  (N, 1, 28, 28)
    Output: (N, 10)
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DigitCNN()
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")

    # Verify output shape
    dummy = torch.zeros(8, 1, 28, 28)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")   # (8, 10)
    assert out.shape == (8, 10), "Output shape mismatch!"
    print("Shape check passed.")
