# DL-Eng: Deep Learning Research & Engineering

`dl-eng` is a deep learning research and engineering monorepo organized around a shared library and independent research projects. It targets modern deep learning systems such as transformers, diffusion models, representation learning, and broader generative modeling workflows. The goal is to provide minimum, extensible abstractions for models, learners, and inference — bridging research prototyping and production-ready engineering.

---

## Project Architecture

```text
dl-eng/
├── dl_eng/                     # Shared library package (pip install -e .)
│   ├── interfaces/             # Contracts between subsystems
│   ├── models/                 # Model families and builders
│   ├── learners/               # Optimization and training logic
│   ├── inference/              # Sampling and inference orchestration
│   └── data/                   # Batches, datasets, and loaders
├── projects/                   # One subdirectory per research project
│   └── <project>/              # e.g., linear_regression
│       ├── <project>/          # Project package (pip install -e projects/<project>)
│       │   ├── data.py
│       │   ├── model.py
│       │   └── ...
│       ├── configs/
│       │   └── config.yaml     # Hyperparameters and paths
│       ├── runs/               # Git-ignored; run outputs land here
│       │   └── <run_id>/       # e.g., {name}_{yyyymmdd}_{timestamp}_s{seed}_g{git_hash}
│       │       ├── config.yaml
│       │       ├── train_metrics.csv
│       │       ├── eval_metrics.csv
│       │       ├── train_curve.png
│       │       ├── eval_curve.png
│       │       └── checkpoints/
│       ├── train.py            # Training entrypoint
│       ├── eval.py             # Evaluation entrypoint
│       ├── pyproject.toml
│       └── README.md
├── exports/                    # Frozen, versioned model releases
│   └── <project_v0.x>/         # e.g., diffusion_v0.1
│       ├── config.yaml
│       ├── export_metadata.yaml
│       └── checkpoints/
├── scripts/                    # Utility scripts (promotion, plotting)
├── tests/                      # Integration and unit tests
├── notebooks/                  # Exploratory notebooks
├── pyproject.toml
└── README.md
```

### Mental Model

```text
                ┌──────────────────┐
                │    projects      │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │    learners      │  training loop + metrics
                └────────┬─────────┘
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
        data           models       inference
          ↓              ↓              ↓
                  interfaces / config
```

## Quick Start

**Lightning AI Studio** (conda base environment active by default):
```bash
git clone https://github.com/bowen0701/dl-eng.git && cd dl-eng && make install
```

**Local development** (activate a virtual environment or conda environment first):
```bash
git clone https://github.com/bowen0701/dl-eng.git
cd dl-eng
python3 -m venv .venv && source .venv/bin/activate
make install
```

`make install` installs the shared `dl_eng` library and all project packages found under `projects/`.

## Makefile Targets

| Target           | Description                     |
|------------------|---------------------------------|
| `make install`   | Install all packages            |
| `make test`      | Run test suite                  |
| `make lint`      | Lint and auto-fix with ruff     |
| `make format`    | Format with ruff                |
| `make typecheck` | Type-check `dl_eng/` with mypy  |

## Running a Project

```bash
python projects/<project>/train.py
python projects/<project>/eval.py
```

Training writes per-run outputs under `projects/<project>/runs/<run_id>/`, including `config.yaml`, `train_metrics.csv`, `eval_metrics.csv`, `train_curve.png`, `eval_curve.png`, and `checkpoints/`.

## Promoting to Exports

Once a run is ready for reuse, promote it into the exports bucket:
```bash
python scripts/promote_run_to_export.py --run_id <run_id> --version 0.1
```
Artifacts are stored in `exports/<project_v0.x>/`.

## Engineering Standards
- **Linting / Formatting**: Managed via `ruff`.
- **Configuration**: Type-safe configs using `dataclasses`.
- **Reproducibility**: Every run is stamped with date, timestamp, seed, and git hash.
- **Naming**: Prefer explicit, qualified names (e.g., `tests/test_diffusion_learner.py` over `test_learner.py`).

## Roadmap
- TBD
