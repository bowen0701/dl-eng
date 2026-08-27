import numpy as np

import torch
import torch.nn as nn
from torch import Tensor


def mse_loss(y: np.ndarray, y_hat: np.ndarray) -> float:
    return np.mean((y - y_hat) ** 2)

def linear_regression_numpy(
    X: np.ndarray,  # (N, D)
    y: np.ndarray,  # (N,)
    lr: float = 0.01, 
    epochs: int = 10_000
) -> tuple[np.ndarray, float]:
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0

    for _ in range(epochs):
        # Forward.
        y_hat = X @ w + b  # (N,)
        loss = mse_loss(y, y_hat)

        # Backward: dL/dw = (2/N) X^T (y_hat - y), dL/db = (2/N) sum(y_hat - y)
        err = y_hat - y
        dw = (2 / N) * X.T @ err
        db = (2 / N) * np.sum(err)

        # Update.
        w -= lr * dw
        b -= lr * db

    return w, b


class LinearRegression(nn.Module):
    """PyTorch implementation of Linear Regression."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        # Linear regression.
        self.fc1 = nn.Linear(self.in_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        return x
