# Repository Guidelines

## Project Structure & Module Organization
`dl_eng/` contains the core Python package.

The project follows an **Experiment-First** philosophy:
- `experiments/`: Encapsulated experiment folders. Each contains its own `config.yaml`, (optional) `main.py`, and an `outputs/` subdirectory for runtime data.
- `dl_eng/interfaces/`: abstract contracts between subsystems.
- `dl_eng/models/`: model-family modules and builders.
- `dl_eng/learners/`: optimization and training primitives.
- `dl_eng/inference/`: sampling and inference orchestration.
- `dl_eng/data/`: batch specs, datasets, and loading primitives.
- `dl_eng/config.py`: consolidated `Config` object for parameters and path orchestration.

Other key directories:
- `artifacts/`: promoted and reusable outputs only (in `exports/`).
- `notebooks/`: exploration and analysis.
- `scripts/`: common orchestration entrypoints.
- `tests/`: project tests mirroring the main package structure.

## Build, Test, and Development Commands
Install locally with `pip3 install -e .`.

- `python3 -m pytest tests` runs the full test suite.
- `python3 -m ruff check .` validates lint rules.
- `python3 -m ruff format .` formats the repository.
- `python3 -m mypy dl_eng tests` runs configured type checks.

## Coding Style & Naming Conventions
Use Python 3.9+, 4-space indentation, and double quotes. Ruff enforces import ordering, lint rules, and Google-style docstrings; keep line length within 127 characters.

When using `@abstractmethod` in classes inheriting from `abc.ABC`, use `pass` instead of `raise NotImplementedError`.

Prefer explicit names that encode scope, model family, or task, such as `transformer.py`, `diffusion.py`, and `test_training_loops.py`.

## Testing Guidelines
Write tests with `pytest` in `tests/`, naming files `test_<area>.py` and test functions `test_<behavior>()`. Keep tests close to the module they validate and update tests whenever interfaces or package wiring changes.

## Collaboration Defaults
- For follow-up collaborations, do not add or modify implementation code unless explicitly requested.
- Prioritize design discussions, architectural reviews, and implementation planning to help build design and implementation skills.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects with focused scope. PRs should include a brief description, exact verification commands, and notes on any experiment config or artifact layout changes.
