"""Tests for linear_regression NumPy and PyTorch implementations.

Usage:
    pytest tests/test_linear_regression.py -v
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dl_eng.data.preprocess import train_test_split
from dl_eng.learners.nano_trainer import fit
from dl_eng.models.linear_regression import LinearRegression, linear_regression_numpy

W_TRUE = np.array([2.0, -1.0, 0.5])
B_TRUE = 3.0


@pytest.fixture
def synthetic_data():
    """Synthetic dataset with known ground truth: y = X @ w + b + noise."""
    rng = np.random.default_rng(42)
    N, D = 1000, 3

    X = rng.standard_normal((N, D))
    y = X @ W_TRUE + B_TRUE + rng.normal(0, 0.1, size=N)

    X_train, y_train, X_val, y_val = train_test_split(X, y, val_ratio=0.2)
    return X_train, y_train, X_val, y_val


# ------------------------------------------------------------------
# NumPy
# ------------------------------------------------------------------


def test_numpy_recovers_weights(synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    w, b = linear_regression_numpy(X_train, y_train, lr=0.01, epochs=10_000)

    print(f"Estimated w: {w}, \nb: {b}")
    np.testing.assert_allclose(w, W_TRUE, atol=0.1)
    np.testing.assert_allclose(b, B_TRUE, atol=0.1)


# ------------------------------------------------------------------
# PyTorch
# ------------------------------------------------------------------


def test_pytorch_recovers_weights(synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)  # (N, 1)
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float().unsqueeze(1)

    torch.manual_seed(42)
    model = LinearRegression(in_dim=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)

    fit(model, train_loader, val_loader, loss_fn, optimizer, n_epochs=200)

    w_learned = model.fc1.weight.detach().numpy().flatten()
    b_learned = model.fc1.bias.detach().numpy().item()

    np.testing.assert_allclose(w_learned, W_TRUE, atol=0.2)
    np.testing.assert_allclose(b_learned, B_TRUE, atol=0.2)


