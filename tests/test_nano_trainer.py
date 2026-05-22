"""Tests for nano_trainer functional training loop.

Usage:
    pytest tests/test_nano_trainer.py -v
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dl_eng.learners.nano_trainer import _train, _validate, fit, test

_TRUE_W = torch.tensor([[1.0], [2.0], [-1.0], [0.5]])


@pytest.fixture
def tiny_model():
    """Small linear model for fast testing."""
    torch.manual_seed(42)
    return nn.Linear(4, 1)


@pytest.fixture
def train_loader():
    """20-sample training loader with a fixed linear relationship."""
    torch.manual_seed(42)
    X = torch.randn(20, 4)
    y = X @ _TRUE_W
    return DataLoader(TensorDataset(X, y), batch_size=5, shuffle=False)


@pytest.fixture
def val_loader():
    """10-sample validation loader with a fixed linear relationship."""
    torch.manual_seed(0)
    X = torch.randn(10, 4)
    y = X @ _TRUE_W
    return DataLoader(TensorDataset(X, y), batch_size=5, shuffle=False)


@pytest.fixture
def loss_fn():
    """MSE loss."""
    return nn.MSELoss()


@pytest.fixture
def optimizer(tiny_model):
    """SGD optimizer for tiny_model."""
    return torch.optim.SGD(tiny_model.parameters(), lr=0.01)


# _train


def test_train_returns_float(tiny_model, train_loader, loss_fn, optimizer):
    """_train should return a float epoch average loss."""
    result = _train(tiny_model, train_loader, loss_fn, optimizer)
    assert isinstance(result, float)


def test_train_loss_decreases(tiny_model, train_loader, loss_fn, optimizer):
    """Loss should decrease after sufficient training steps."""
    loss_before = _train(tiny_model, train_loader, loss_fn, optimizer)
    for _ in range(20):
        _train(tiny_model, train_loader, loss_fn, optimizer)
    loss_after = _train(tiny_model, train_loader, loss_fn, optimizer)
    assert loss_after < loss_before


def test_train_empty_loader_returns_zero(tiny_model, loss_fn, optimizer):
    """_train on empty loader should return 0.0."""
    empty = DataLoader(TensorDataset(torch.randn(0, 4), torch.randn(0, 1)))
    assert _train(tiny_model, empty, loss_fn, optimizer) == 0.0


def test_train_logs_batch_loss(
    tiny_model, train_loader, loss_fn, optimizer, capsys
):
    """_train should print batch loss every log_n_steps."""
    _train(tiny_model, train_loader, loss_fn, optimizer, log_n_steps=2)
    assert "batch loss" in capsys.readouterr().out


# _validate


def test_validate_returns_float(tiny_model, val_loader, loss_fn):
    """_validate should return a float epoch average loss."""
    assert isinstance(_validate(tiny_model, val_loader, loss_fn), float)


def test_validate_no_grad(tiny_model, val_loader, loss_fn):
    """_validate should not accumulate gradients."""
    for p in tiny_model.parameters():
        p.grad = None
    _validate(tiny_model, val_loader, loss_fn)
    for p in tiny_model.parameters():
        assert p.grad is None


def test_validate_empty_loader_returns_zero(tiny_model, loss_fn):
    """_validate on empty loader should return 0.0."""
    empty = DataLoader(TensorDataset(torch.randn(0, 4), torch.randn(0, 1)))
    assert _validate(tiny_model, empty, loss_fn) == 0.0


# test


def test_test_returns_float(tiny_model, val_loader, loss_fn):
    """Test should return a float average loss."""
    assert isinstance(test(tiny_model, val_loader, loss_fn), float)


def test_test_consistent_with_validate(tiny_model, val_loader, loss_fn):
    """Test and _validate should return identical loss on the same loader."""
    val_loss = _validate(tiny_model, val_loader, loss_fn)
    test_loss = test(tiny_model, val_loader, loss_fn)
    assert abs(val_loss - test_loss) < 1e-6


def test_test_empty_loader_returns_zero(tiny_model, loss_fn):
    """Test on empty loader should return 0.0."""
    empty = DataLoader(TensorDataset(torch.randn(0, 4), torch.randn(0, 1)))
    assert test(tiny_model, empty, loss_fn) == 0.0


# fit


def test_fit_runs(tiny_model, train_loader, val_loader, loss_fn, optimizer):
    """Fit should complete without error."""
    fit(tiny_model, train_loader, val_loader, loss_fn, optimizer, n_epochs=2)


def test_fit_model_on_device(
    tiny_model, train_loader, val_loader, loss_fn, optimizer
):
    """Fit should move model to specified device."""
    fit(tiny_model, train_loader, val_loader, loss_fn, optimizer, device="cpu")
    for p in tiny_model.parameters():
        assert p.device.type == "cpu"


def test_fit_logs_epoch_loss(
    tiny_model, train_loader, val_loader, loss_fn, optimizer, capsys
):
    """Fit should print train_loss and val_loss for each epoch."""
    fit(tiny_model, train_loader, val_loader, loss_fn, optimizer, n_epochs=2)
    captured = capsys.readouterr()
    assert "train_loss" in captured.out
    assert "val_loss" in captured.out
    assert captured.out.count("epoch") == 2
