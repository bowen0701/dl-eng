import numpy as np

import torch
import torch.nn as nn
from torch import Tensor


def softmax(Z: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_Z = np.exp(Z)
    return exp_Z / exp_Z.sum(axis=axis, keepdims=True)


def stable_softmax(Z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Z: (N, K) logits -> (N, K) probabilities."""
    max_Z = Z.max(axis=axis, keepdims=True)
    exp_Z = np.exp(Z - max_Z)
    return exp_Z / exp_Z.sum(axis=axis, keepdims=True)


def stable_logsumexp(Z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Z: (N, K) -> (N,). log(sum(exp(Z))) via max-shift."""
    max_Z = Z.max(axis=axis, keepdims=True)           # (N, 1)
    max_Z = np.where(np.isfinite(max_Z), max_Z, 0.0)  # guard fully-masked rows
    return max_Z.squeeze(axis) + np.log(np.sum(np.exp(Z - max_Z), axis=axis))  # (N,)


def stable_log_softmax(Z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Z: (N, K). log_softmax = Z - logsumexp(Z)."""
    lse = stable_logsumexp(Z, axis=axis)       # (N,)
    return Z - np.expand_dims(lse, axis=axis)  # (N, K)


def stable_cross_entropy(Z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Fused softmax + CE: forward loss and backward gradient.
    Z: (N, K) logits, y: (N,) integer labels.
    Returns (scalar loss, (N, K) gradient w.r.t. Z).
    """
    N = Z.shape[0]
    logp = stable_log_softmax(Z, axis=-1)  # (N, K)
    loss = -logp[np.arange(N), y].mean()   # scalar

    # Backward: dL/dZ = (softmax(Z) - onehot(y)) / N
    dZ = np.exp(logp)                      # (N, K), recover softmax
    dZ[np.arange(N), y] -= 1.0             # (N, K), substract onehot(y)
    dZ /= N
    return loss, dZ


def softmax_regression_numpy(
    X: np.ndarray, y: np.ndarray, n_classes: int, 
    lr: float = 0.01, epochs: int = 100,
) -> tuple[np.ndarray, float]:
    """Softmax Regression implementation on NumPy.
    
    X: (N, D) feature matrix, y: (N,) integer labels in {0, ..., n_classes - 1}
    Reference: https://bowen0701.github.io/re-log/manual-layers/
    """
    N, D = X.shape
    W = np.zeros((D, n_classes))                # (D, K)
    b = np.zeros(n_classes)                     # (K,)

    # Forward + backward passes (fused in _stable_cross_entropy)
    for _ in range(epochs):
        Z = X @ W + b                           # (N, K)
        loss, dZ = stable_cross_entropy(Z, y)  # scalar, (N, K) 

        # Gradients w.r.t. weights and biases
        dW = X.T @ dZ                           # (D, K)
        db = dZ.sum(axis=0)                     # (K,)

        # Update
        W -= lr * dW
        b -= lr * db

    return W, b


class SoftmaxRegression(nn.Module):
    """Softmax Regression implementation on PyTorch."""
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Softmax regression: return raw logits, pair with nn.CrossEntropyLoss.
        self.fc1 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc1(x)
