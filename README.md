# TRAXIS

Research tools for reconstructing backside material loss in aluminum plates
from a fixed PZT network. The target is depth in millimeters, not a damage class.

This repository is under development. Neural candidates have received only
brief, bounded debug updates; no research training cycle has run. No simulation
has been validated against a physical plate. Passing software checks do not
establish physical accuracy or a superior model.

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
.\.venv\Scripts\python.exe -m traxis dispersion --output outputs/dispersion.json
.\.venv\Scripts\python.exe -m traxis verify --output outputs/verification
```

Verification is CPU-only and bounded. There is no automatic training or large
dataset generation command. Runtime limits, assumptions, and current work are
tracked in [the implementation checklist](docs/implementation-status.md).

The `dispersion` command computes fundamental A0/S0 phase speed, group speed
and wavelength using an independent Rayleigh-Lamb traction determinant. Use
group speed for packet delays and phase wavelength when considering spatial
resolution. It covers a documented low-frequency range and uses nominal
material constants; it is not an intact physical-plate validation result.

## Waveform data

The [data contract](docs/data-contract.md) describes versioned waveform banks,
specimen-level split checks and partial-observation replay. Import explicitly
mapped, aligned current/control scalar traces with:

```powershell
.\.venv\Scripts\python.exe -m traxis import-traces --manifest bundle/manifest.json --output data/scan.npz
```

DAS returns a localization intensity in signal units. It must be calibrated
before it can serve as a quantitative depth baseline. Analytic point-scattering
test fixtures are software checks, not an elastic simulation dataset.

## Reconstruction candidates

Model A combines a waveform CNN, spatial attention and a convolutional map
decoder. Model B combines DAS with a residual U-Net and waveform context.
Both support 128/256 outputs and extra decoder blocks. See
[model details and checks](docs/reconstruction-models.md).

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-learning.txt
.\.venv\Scripts\python.exe -m traxis verify-models
```

This runs exactly eight single-step CPU debug configurations, with resource
limits. The optional model tests are skipped if PyTorch is absent; running
the core suite alone does not validate the neural candidates.

## Experiment runner

The [experiment workflow](docs/experiment-workflow.md) covers dataset indexing,
bounded subset training/resume, training-only DAS depth calibration, ensembles,
physical adaptation roles and frozen equal-budget replay. Start with the
debug configurations under `configs/`; these perform a few updates and do
not provide a research training schedule or validated physical parameters.

For a monitored cache-to-replay software pilot using analytic fixtures:

```powershell
.\.venv\Scripts\python.exe -m traxis verify-workflow
```

It performs four total debug updates and checks all three reconstruction
methods under matched fixed/random record budgets, plus ensemble inference.
The generated results remain explicitly debug-only.

Adaptive policies include a masked RL actor/value selector and a distinct
uncertainty-path heuristic. [Adaptive sensing details](docs/adaptive-sensing.md)
describe their public inputs, reward, provenance checks and current limits.
The `train-policy` and `verify-policy` commands support bounded policy updates
and a monitored fixture pilot; the latter requires the existing debug dataset
and reconstruction ensemble described in that document.

Reproducible virtual specimen recipes, optional CAD-conforming recess meshing
and bounded local elastic waveform banks are described in
[scenario generation](docs/scenario-generation.md). Their data remains marked
as pilot-only until forward-physics and physical validation pass.

`plan-scenarios` writes unsolved study recipes; `export-solver-request` compiles
one reference task with anchored geometry labels and sample times.
`verify-forward` runs a short CAD-to-elastic-bank pilot without model training.

[Finite-plate waveform refinement](docs/waveform-validation.md) measures mesh
and time-step errors independently and exports checked trace plots. The first
completed sweep passed its temporal screen but failed its spatial screen;
local mechanical banks remain unsuitable for research training.

An optional [Elmer reference check](docs/elmer-reference.md) compares independent
3D stiffness/consistent-mass assembly on rectangular and skewed meshes. A
portable Windows solver trial runs entirely inside the project; matching
coarse operators does not establish converged finite-plate waveforms.

[Quadratic guided-mode refinement](docs/elmer-waveguide.md) checks Elmer's
P2 modes against Rayleigh-Lamb roots with separate in-plane/thickness axes.
Small analytic errors support further reference work; one strict decrease
criterion failed and remains recorded.

[Independent transient checks](docs/elmer-transient.md) verify volume/nodal
force units, named probes and time refinement on fixed P1/P2 systems. Passing
these checks does not establish finite-plate spatial or PZT accuracy.

[Quadratic apertures](docs/quadratic-apertures.md) preserve circular area,
centroid and second moments and pass native burst/reciprocity checks. They
provide a P2 mechanical reference interface; local bank generation remains P1.

[Bounded P2 waveform references](docs/elmer-waveforms.md) combine independent
Elmer sparse operators with checked implicit time integration. A small finite
plate ran under the resource guard; spatial/time and PZT acceptance remain
open, and these traces are not approved research training data.

The [scientific reporting command](docs/scientific-reporting.md) exports
error-versus-record curves, depth/error/uncertainty maps, 2.5D surfaces,
parameter counts and compute tables from completed frozen replay results.
Figures retain physical units and debug status; reporting performs no training.

The [reserved-parent calibration workflow](docs/uncertainty-calibration.md)
fits and assesses post-hoc prediction intervals for a frozen ensemble and
inspection protocol. It requires separate calibration plates and matching
distribution assumptions; the current debug models do not produce useful
research uncertainty estimates.

[Matched studies](docs/matched-studies.md) plan architecture/resolution/decoder
ablations with paired seeds, bounded job advances, checkpoint ownership and
exact observation-exposure checks. Freezing writes isolated comparison and
ensemble plans without launching their evaluation.

