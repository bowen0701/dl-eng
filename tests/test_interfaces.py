"""Interface, model spec, learner state, and inference config smoke tests.

Usage:
    pytest tests/test_interfaces.py -v
"""

from dl_eng.interfaces import (
    DatasetInterface,
    InferenceInterface,
    LearnerInterface,
    ModelInterface,
)
from dl_eng.models import DiffusionModelSpec, TransformerModelSpec
from dl_eng.learners import LearnerState
from dl_eng.inference import SamplingConfig


# ------------------------------------------------------------------
# Interfaces
# ------------------------------------------------------------------


def test_interfaces_are_importable() -> None:
    """Core interfaces should be importable from the package root."""
    assert DatasetInterface is not None
    assert InferenceInterface is not None
    assert LearnerInterface is not None
    assert ModelInterface is not None


# ------------------------------------------------------------------
# Model specs
# ------------------------------------------------------------------


def test_transformer_model_spec_defaults() -> None:
    """Transformer defaults should be populated."""
    spec = TransformerModelSpec()

    assert spec.d_model > 0
    assert spec.n_layers > 0


def test_diffusion_model_spec_defaults() -> None:
    """Diffusion defaults should be populated."""
    spec = DiffusionModelSpec()

    assert spec.timesteps > 0
    assert spec.prediction_target == "epsilon"


# ------------------------------------------------------------------
# Learner state & inference config
# ------------------------------------------------------------------


def test_learner_state_defaults() -> None:
    """Learner state should expose simple training progress fields."""
    state = LearnerState()

    assert state.step == 0
    assert state.epoch == 0


def test_sampling_config_defaults() -> None:
    """Sampling config should expose generation-friendly defaults."""
    config = SamplingConfig()

    assert config.temperature == 1.0
