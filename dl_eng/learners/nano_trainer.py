import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _train(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    log_n_steps: int = 100,
    device: str | torch.device = "cpu",
) -> float:
    """Run one training epoch, return average loss."""
    model.train()

    total_loss = 0.0
    n_samples = 0

    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss.
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_loss = loss.item()
        total_loss += batch_loss * len(y)
        n_samples += len(y)

        if (batch + 1) % log_n_steps == 0:
            print(f"step {batch + 1}/{len(train_loader)} batch loss: {batch_loss:.4f}")

    if n_samples == 0:
        return 0.0

    return total_loss / n_samples


def _validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: str | torch.device = "cpu",
) -> float:
    """Run one validation epoch, return average loss."""
    model.eval()

    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)

            # Compute prediction and loss.
            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item() * len(y)
            n_samples += len(y)

    if n_samples == 0:
        return 0.0

    return total_loss / n_samples


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 1,
    log_n_steps: int = 100,
    device: str | torch.device = "cpu",
) -> None:
    """Train and validate model for n_epochs, logging loss each epoch.

    Args:
        model: The neural network to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        loss_fn: Loss function (e.g. nn.CrossEntropyLoss()).
        optimizer: Optimizer (e.g. torch.optim.Adam).
        n_epochs: Number of full passes over the training data.
        log_n_steps: Log training batch loss every this many steps.
        device: Device to run on, e.g. "cpu" or "cuda".

    Usage:
        model = nn.Linear(16, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        fit(model, train_loader, val_loader, loss_fn, optimizer, n_epochs=10, device="cuda")
    """
    model = model.to(device)

    for epoch in range(n_epochs):
        train_loss = _train(model, train_loader, loss_fn, optimizer, log_n_steps, device)  # noqa: E501
        val_loss = _validate(model, val_loader, loss_fn, device)
        print(f"epoch {epoch + 1}/{n_epochs} train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")  # noqa: E501


def test(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: str | torch.device = "cpu",
) -> float:
    """Evaluate model on test set, return average loss.

    Args:
        model: Trained neural network to evaluate.
        test_loader: DataLoader for test data.
        loss_fn: Loss function (e.g. nn.CrossEntropyLoss()).
        device: Device to run on, e.g. "cpu" or "cuda".

    Returns:
        Average loss over the test set.

    Usage:
        loss = test(model, test_loader, nn.CrossEntropyLoss(), device="cuda")
        print(f"Test loss: {loss:.4f}")
    """
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            # Compute prediction and loss.
            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item() * len(y)
            n_samples += len(y)

    if n_samples == 0:
        return 0.0
    return total_loss / n_samples
