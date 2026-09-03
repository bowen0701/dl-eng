"""Tests for softmax_regression NumPy and PyTorch implementations.

Softmax weight identifiability issue:
    softmax(z) = softmax(z + c) for any constant c, so the MLE solution is
    only unique up to an additive constant across classes. Naive weight
    comparison (assert_allclose(W, W_TRUE)) fails even when the model has
    fully converged. Fix: center both learned and true weights by subtracting
    the class-mean before comparing. See _center() below.

Usage:
    pytest tests/test_softmax_regression.py -v
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dl_eng.data.preprocess import train_test_split
from dl_eng.learners.nano_trainer import fit
from dl_eng.models.softmax_regression import SoftmaxRegression, softmax_regression_numpy, stable_softmax

N_CLASSES = 3
W_TRUE = np.array([
    [2.0, -1.0, 0.5],   # class 0
    [-1.0, 2.0, -0.5],  # class 1
    [0.0, -0.5, 1.5],   # class 2
])  # (K, D)
B_TRUE = np.array([0.3, -0.2, 0.1])  # (K,)


@pytest.fixture
def synthetic_data():
    """Synthetic multi-class dataset with probabilistic labels: y ~ Categorical(softmax(X @ W^T + b))."""
    rng = np.random.default_rng(42)
    N, D = 5000, 3

    X = rng.standard_normal((N, D))
    logits = X @ W_TRUE.T + B_TRUE  # (N, K)
    probs = stable_softmax(logits, axis=-1)  # (N, K)
    y = np.array([rng.choice(N_CLASSES, p=p) for p in probs])  # (N,)

    X_train, y_train, X_val, y_val = train_test_split(X, y, val_ratio=0.2)
    return X_train, y_train, X_val, y_val


# ------------------------------------------------------------------
# NumPy
# ------------------------------------------------------------------


def _center(W, b):
    """Remove softmax's additive ambiguity: softmax(z) = softmax(z + c)."""
    w_mean = W.mean(axis=-1, keepdims=True)  # mean across classes
    return W - w_mean, b - b.mean()


def test_numpy_recovers_weights(synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    W, b = softmax_regression_numpy(X_train, y_train, n_classes=N_CLASSES, lr=0.1, epochs=2000)

    # W is (D, K), W_TRUE is (K, D): compare centered to remove additive ambiguity
    W_c, b_c = _center(W, b)
    W_true_c, b_true_c = _center(W_TRUE.T, B_TRUE)
    print(f"Centered W:\n{W_c.T}, \nb: {b_c}")
    np.testing.assert_allclose(W_c, W_true_c, atol=0.3)
    np.testing.assert_allclose(b_c, b_true_c, atol=0.3)


# ------------------------------------------------------------------
# PyTorch
# ------------------------------------------------------------------


def test_pytorch_recovers_weights(synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    torch.manual_seed(42)
    model = SoftmaxRegression(input_dim=3, output_dim=N_CLASSES)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)

    fit(model, train_loader, val_loader, optimizer, loss_fn, n_epochs=200)

    # fc1.weight is (K, D), matching W_TRUE layout: compare centered
    W = model.fc1.weight.detach().numpy()
    b = model.fc1.bias.detach().numpy()

    W_c, b_c = _center(W.T, b)
    W_true_c, b_true_c = _center(W_TRUE.T, B_TRUE)
    print(f"Centered W:\n{W_c.T}, \nb: {b_c}")
    np.testing.assert_allclose(W_c, W_true_c, atol=0.3)
    np.testing.assert_allclose(b_c, b_true_c, atol=0.3)
