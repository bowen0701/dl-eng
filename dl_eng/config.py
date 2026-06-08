"""Shared configuration objects for runs and promoted artifacts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class Config:
    """Consolidated experiment configuration and filesystem orchestration."""

    name: str  # e.g., "transformers/baseline_v1"
    seed: int = 42
    model_family: str = "transformer"
    task: str = "pretraining"
    output_root: Path = Path("projects")
    artifact_root: Path = Path("artifacts")
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def experiment_dir(self) -> Path:
        """Filesystem location for this specific experiment."""
        return self.output_root / self.name

    @property
    def run_dir(self) -> Path:
        """Filesystem location for runtime outputs (metrics, plots)."""
        return self.experiment_dir / "outputs"

    @property
    def metrics_path(self) -> Path:
        """Path to the metrics CSV file."""
        return self.run_dir / "metrics.csv"

    @property
    def eval_path(self) -> Path:
        """Path to the evaluation CSV file."""
        return self.run_dir / "eval.csv"

    @property
    def learning_curve_path(self) -> Path:
        """Path to the learning curve plot."""
        return self.run_dir / "learning_curve.png"

    @property
    def eval_curve_path(self) -> Path:
        """Path to the evaluation curve plot."""
        return self.run_dir / "eval_curve.png"

    @property
    def tensorboard_dir(self) -> Path:
        """Path to the TensorBoard logs."""
        return self.run_dir / "tensorboard"

    @property
    def export_dir(self) -> Path:
        """Filesystem location for promoted artifacts."""
        return self.artifact_root / "exports"
