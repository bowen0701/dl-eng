import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable: avoid overflow in exp(-z) for large positive z
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def naive_bce_loss(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, np.ndarray]:
    """BCE from probabilities. Uses epsilon to guard log(0).

    y: (N,) binary labels, y_hat: (N,) predicted probabilities (sigmoid output).
    """
    N = y.shape[0]
    eps = 1e-8
    loss = - np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))  # scalar
    dz = (y_hat - y) / N  # (N,)
    return loss, dz


def stable_bce_loss(z: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Fused sigmoid + BCE: forward loss and backward gradient.

    z: (N,) logits, y: (N,) binary labels.
    L = max(z,0) - zy + log(1 + exp(-|z|)).
    """
    N = z.shape[0]
    loss = np.mean(np.maximum(z, 0) - z * y + np.log(1 + np.exp(-np.abs(z))))
    dz = (stable_sigmoid(z) - y) / N
    return loss, dz


def logistic_regression_numpy(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 100) -> tuple[np.ndarray, float]:
    """Logistic Regression implementation on NumPy.

    X: (N, D), y: (N,)
    Reference: https://bowen0701.github.io/re-log/manual-layers/
    """
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0

    for _ in range(epochs):
        # Forward + backward passes (fused in stable_bce_loss)
        z = X @ w + b                     # (N,)
        loss, dz = stable_bce_loss(z, y)  # scalar, (N,)

        # Gradients w.r.t. weights and bias
        dw = X.T @ dz                     # (D,)
        db = np.sum(dz)                   # scalar

        # Update
        w -= lr * dw
        b -= lr * db

    return w, b


class LogisticRegression(nn.Module):
    """PyTorch implementation of Logistic Regression."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        # Logistic regression.
        self.fc1 = nn.Linear(self.in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
