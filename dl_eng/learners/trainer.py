from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dl_eng.learners.logging import TrainerLogger


class Trainer:
    """Standard trainer for deep learning models.

    This trainer uses a TrainerLogger to decouple observation from optimization.
    """

    def _train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        logger: TrainerLogger,
        global_step: int,
        log_n_steps: int = 50,
        device: str | torch.device = "cpu",
    ) -> tuple[float, int]:
        """Run one training epoch, return average loss and updated global step."""
        model.train()

        # Epoch-level and window-level accumulators initialized as paired units
        total_loss, total_samples = 0.0, 0
        window_loss, window_samples = 0.0, 0

        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            # Compute prediction and loss.
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backprop.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_size = len(y)
            # Use sample-weighted loss for maximum precision
            weighted_loss = loss.item() * batch_size

            total_loss += weighted_loss
            total_samples += batch_size

            window_loss += weighted_loss
            window_samples += batch_size
            global_step += 1

            if (batch + 1) % log_n_steps == 0:
                # Log precise mean for the window
                avg_window_loss = window_loss / window_samples
                logger.on_batch(global_step, avg_window_loss)
                # Reset window accumulators using idiomatic tuple assignment
                window_loss, window_samples = 0.0, 0

        # Flush any remaining samples in the final window
        if window_samples > 0:
            logger.on_batch(global_step, window_loss / window_samples)

        avg_epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_epoch_loss, global_step

    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: str | torch.device = "cpu",
    ) -> float:
        """Run one validation epoch, return average loss."""
        model.eval()

        total_loss, total_samples = 0.0, 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                # Compute prediction and loss.
                pred = model(X)
                loss = loss_fn(pred, y)

                total_loss += loss.item() * len(y)
                total_samples += len(y)

        return total_loss / total_samples if total_samples > 0 else 0.0

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        n_epochs: int = 1,
        log_n_steps: int = 50,
        device: str | torch.device = "cpu",
        logger: Optional[TrainerLogger] = None,
    ) -> None:
        """Train and validate model for n_epochs.

        Args:
            model: The neural network to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            loss_fn: Loss function (e.g. nn.CrossEntropyLoss()).
            optimizer: Optimizer (e.g. torch.optim.Adam).
            n_epochs: Number of full passes over the training data.
            log_n_steps: Log training batch loss every this many steps.
            device: Device to run on, e.g. "cpu" or "cuda".
            logger: Optional TrainerLogger for metrics and progress.

        Usage:
            from torch.utils.tensorboard import SummaryWriter
            from dl_eng.learners.logging import TrainerLogger

            writer = SummaryWriter(log_dir="runs/exp1")
            logger = TrainerLogger(writer)

            model = nn.Linear(16, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            
            trainer = Trainer()
            trainer.fit(model, train_loader, val_loader, loss_fn, optimizer, logger=logger)
        """
        model = model.to(device)

        # Ensure we have a logger, even if it's a "silent" one (no writer)
        if logger is None:
            logger = TrainerLogger(writer=None)

        global_step = 0
        for epoch in range(n_epochs):
            train_loss, global_step = self._train(
                model, train_loader, loss_fn, optimizer, logger, global_step, log_n_steps, device
            )
            val_loss = self._validate(model, val_loader, loss_fn, device)

            logger.on_epoch(epoch, train_loss, val_loss)

    def test(
        self,
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
            trainer = Trainer()
            test_loss = trainer.test(model, test_loader, loss_fn)
        """
        model = model.to(device)
        model.eval()

        total_loss, total_samples = 0.0, 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)

                # Compute prediction and loss.
                pred = model(X)
                loss = loss_fn(pred, y)

                total_loss += loss.item() * len(y)
                total_samples += len(y)

        return total_loss / total_samples if total_samples > 0 else 0.0
