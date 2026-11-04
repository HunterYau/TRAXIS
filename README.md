# TRAXIS

Research tools for reconstructing backside material loss in aluminum plates
from a fixed PZT network. The target is depth in millimeters, not a damage class.

This repository is under development. No model has been trained, and no
simulation has been validated against a physical plate. Do not use a passing
unit test as evidence of physical accuracy.

## Solver decision

Use COMSOL's 3D Elastic Waves / Piezoelectric Waves, Time Explicit formulation
as the preferred reference solver if a suitable license is available. Build
the dataset, reconstruction models, and sensing experiments in Python.
See [the solver assessment](docs/solver-assessment.md) for alternatives,
limitations, and the acceptance gates before generating training data.

The local solver is a small, inspectable 3D linear-elastic tetrahedral FEM
backend for numerical verification and pilot experiments. It is not a
replacement for calibrated PZT electromechanics or a validated commercial
solver. Small smoke geometries are deliberately different from the full
500 x 500 x 3 mm research plate.

## Setup

Python 3.12 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m traxis preflight
.\.venv\Scripts\python.exe -m traxis verify --output outputs/verification
```

Verification is CPU-only and bounded. There is no automatic training or large
dataset generation command. Runtime limits, assumptions, and current work are
tracked in [the implementation checklist](docs/implementation-status.md).

