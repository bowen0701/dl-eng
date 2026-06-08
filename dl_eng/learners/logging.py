from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class TrainerLogger:
    """Handles experiment observation, including console output and TensorBoard.

    This class decouples the 'how to log' from the 'how to train' logic in the Trainer.
    """

    def __init__(self, writer: Optional[SummaryWriter] = None):
        """Initialize the logger.

        Args:
            writer: An optional TensorBoard SummaryWriter instance.
        """
        self.writer = writer

    def on_batch(self, step: int, loss: float):
        """Log metrics at the end of a training batch/window.

        Args:
            step: The global training step count.
            loss: The averaged loss over the logging window.
        """
        if self.writer:
            self.writer.add_scalar("Loss/Train_Batch", loss, step)

    def on_epoch(self, epoch: int, train_loss: float, val_loss: float):
        """Log metrics and print progress at the end of an epoch.

        Args:
            epoch: The current epoch index (0-indexed).
            train_loss: The average training loss for the epoch.
            val_loss: The average validation loss for the epoch.
        """
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if self.writer:
            self.writer.add_scalar("Loss/Train_Epoch", train_loss, epoch)
            self.writer.add_scalar("Loss/Val_Epoch", val_loss, epoch)

    def close(self):
        """Flush and close the TensorBoard writer."""
        if self.writer:
            self.writer.close()
