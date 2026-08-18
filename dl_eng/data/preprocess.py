import numpy as np
import torch
from torch import Tensor
from typing import Union

ArrayLike = Union[np.ndarray, Tensor]


# ------------------------------------------------------------------
# NumPy
# ------------------------------------------------------------------


def shuffle_numpy(
    X: np.ndarray,  # X: (N, D)
    y: np.ndarray,  # y: (N,)
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Usage: 
        rng: np.random.Generator = np.random.default_rng(42)
        X, y = shuffle_numpy(X, y, rng)
    """
    N = X.shape[0]

    idx = rng.permutation(N)
    X, y = X[idx], y[idx]
    return X, y


def train_test_split(
    X: ArrayLike,  # X: (N, D)
    y: ArrayLike,  # y: (N,)
    val_ratio: float = 0.2,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    N = X.shape[0]

    split = int(N * (1 - val_ratio))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    return X_train, y_train, X_val, y_val


def normalize_numpy(
    X_train: np.ndarray,  # X_train: (N_train, D)
    X_val: np.ndarray,    # X_val: (N_val, D)
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)      # (D,)
    std = X_train.std(axis=0) + eps  # (D,)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val, mean, std


def preprocess_numpy(
    X: np.ndarray,  # X: (N, D)
    y: np.ndarray,  # y: (N,)
    rng: np.random.Generator,
    val_ratio: float = 0.2, 
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns normalized, split data."""

    # 1. Shuffle
    X, y = shuffle_numpy(X, y, rng)

    # 2. Train/val split
    X_train, y_train, X_val, y_val = train_test_split(X, y, val_ratio=val_ratio)

    # 3. Normalize (fit on train, apply to val)
    X_train, X_val, X_train_mean, X_train_std = normalize_numpy(X_train, X_val, eps=eps)

    return X_train, y_train, X_val, y_val, X_train_mean, X_train_std


# ------------------------------------------------------------------
# PyTorch
# ------------------------------------------------------------------


def shuffle_pytorch(
    X: Tensor,  # X: (N, D)
    y: Tensor,  # y: (N,)
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """
    Usage:
        generator = torch.Generator().manual_seed(42)
        X, y = shuffle_pytorch(X, y, generator)
    """
    N = X.shape[0]

    idx = torch.randperm(N, generator=generator)
    return X[idx], y[idx]


def normalize_pytorch(
    X_train: Tensor,  # X_train: (N_train, D)
    X_val: Tensor,    # X_val: (N_val, D)
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mean = X_train.mean(dim=0)      # (D,)
    std = X_train.std(dim=0) + eps   # (D,)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val, mean, std


def preprocess_pytorch(
    X: Tensor,  # X: (N, D)
    y: Tensor,  # y: (N,)
    generator: torch.Generator,
    val_ratio: float = 0.2,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Returns normalized, split data."""

    # 1. Shuffle
    X, y = shuffle_pytorch(X, y, generator)

    # 2. Train/val split
    X_train, y_train, X_val, y_val = train_test_split(X, y, val_ratio=val_ratio)

    # 3. Normalize (fit on train, apply to val)
    X_train, X_val, X_train_mean, X_train_std = normalize_pytorch(X_train, X_val, eps=eps)

    return X_train, y_train, X_val, y_val, X_train_mean, X_train_std
