"""Configuration tests."""

from pathlib import Path

from dl_eng.config import Config


def test_config_derived_paths() -> None:
    """Config should correctly derive experiment and output paths."""
    config = Config(name="transformers/run_001")

    # Experiment and Run paths
    assert config.experiment_dir == Path("projects/transformers/run_001")
    assert config.run_dir == Path("projects/transformers/run_001/outputs")

    # Nested output paths
    assert config.metrics_path == Path("projects/transformers/run_001/outputs/metrics.csv")
    assert config.eval_path == Path("projects/transformers/run_001/outputs/eval.csv")
    assert config.learning_curve_path == Path("projects/transformers/run_001/outputs/learning_curve.png")
    assert config.eval_curve_path == Path("projects/transformers/run_001/outputs/eval_curve.png")

    # Artifact paths
    assert config.export_dir == Path("artifacts/exports")


def test_config_defaults() -> None:
    """Config should expose stable defaults."""
    config = Config(name="smoke")

    assert config.seed == 42
    assert config.output_root == Path("projects")
    assert config.artifact_root == Path("artifacts")
    assert config.model_family == "transformer"
