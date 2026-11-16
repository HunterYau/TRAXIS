# Coupled XYZ controls and paired assessment

The frozen proposal now has an executable full-window launcher and a
paired intact/recessed/scattering assessor. Q4 levels refine exterior
spacing 5/4.5/4 mm and lower/upper layers (1,1)/(2,2)/(3,3), with the
predeclared angular/radial counts and 491,520 steps over 500 us. The
3,072 MiB workspace request and eight-hour wall allowance retain all
memory/temperature guards. All fourteen older case/screen/profile hashes
and the existing H protocol remain unchanged.

Each XYZ plan snapshots the exact proposal and its loader alongside the
worker, launcher and unchanged independent CPU checker. Worker children
explicitly select their parent's checkout. This fixes embedded Python
runtimes that otherwise load the main checkout through `_pth`, even when
the child runs from a different working directory. An actual child CLI
test from the other checkout verifies module selection.

The CPU wrapper uses the original `verify_embedded_window.py` without
changing its numerical checks, with a 600-second / 4,096 MiB envelope and
89.9% system-RAM ceiling. Assessment requires both its original passing
proof and a passing, source/hash/resource-bound wrapper receipt. A failed
outer check cannot leave a sidecar claiming success. Outputs must be fresh.

The paired assessment requires six distinct XYZ runs and CPU checks, the
same GPU/precision and complete float64 raw grids. It uses exact stride-15
sample selection, subtracts intact from recessed before comparison, and
requires all nine receiver checks to show identifiable decreasing changes
with finest changes below 1%. No fitting, phase adjustment or interpolation
is allowed. XY-only and older time controls cannot substitute for an XYZ
level. A pass would establish this one mechanical spatial screen; broader
frequency, defect, calibrated PZT and physical-data checks still remain.

Forty-four focused checks passed with no skips, including manufactured
signals whose total fields pass while scattering fails, changed grids,
bad CPU/source/resource bindings, and existing-control regression checks.
These unit checks are not physical validation. Six actual small Q4 GPU
fixtures exercised intact/recessed topology at all three thickness-layer
counts. Their force contractions agree with the independent CPU path to
at most 2.006e-16 relative L2, with exact mass arrays. Every planned
491,520-step / three-probe interface reached synchronized batch 16 before
an intentional stop. Fixture sizes were 12,591–40,875 vector DOFs. These
checks do not allocate the full 20-million-DOF operators or complete any
XYZ wave window.

A fresh two-version CUDA transition bundle passes with zero differences.
Historical total/scattering reviews retain the temporal pass and the
receiver-3 spatial-scattering failure of 1.344761%. A separate H-archive
check initially hit the isolated Green helper's input-path guard. Eight
files were then copied unchanged into the isolated output root, with
before/after hashes, and the original guard and source/CPU receipt binding
passed. This is an archive review, not a new trajectory or CPU computation.
Compact bindings are in `evidence/embedded-xyz-controls-debug-20261115.json`.

Run the six controls serially only after the active H window and its CPU
check finish, and after integrating this branch into the selected checkout.
In a normal project Python environment, each level/state uses:

```powershell
python -m examples.embedded_wave_budget --xyz-level 0 --recess --output outputs/xyz-recess-level0
python -m examples.verify_embedded_xyz_window --run outputs/xyz-recess-level0 --output outputs/xyz-recess-level0-cpu
```

Repeat at levels 1 and 2 and without `--recess` for intact outputs. On this
host the root portable interpreter works after integration; running the
isolated checkout still requires its parent-process path bootstrap. The
launcher now supplies the matching bootstrap to its worker automatically.
Keep bound sources fixed through each wave and CPU check. Do not reuse a
294,912-step coarse H window for a 491,520-step XYZ comparison.

After all six complete, the assessment command is:

```powershell
python -m examples.embedded_xyz_study --intact-runs outputs/xyz-intact-level0 outputs/xyz-intact-level1 outputs/xyz-intact-level2 --intact-cpu-checks outputs/xyz-intact-level0-cpu outputs/xyz-intact-level1-cpu outputs/xyz-intact-level2-cpu --recessed-runs outputs/xyz-recess-level0 outputs/xyz-recess-level1 outputs/xyz-recess-level2 --recessed-cpu-checks outputs/xyz-recess-level0-cpu outputs/xyz-recess-level1-cpu outputs/xyz-recess-level2-cpu --output outputs/xyz-assessment
```

If sources changed between completed controls, supply `--source-check`
with a fresh matching transition bundle. The current isolated bundle is
`outputs/embedded-xyz-source-transitions-20261115-v1/source-transitions-check.json`,
fingerprint 5b08ba384f68402a92204e7eb3a6c1e2dcbed91667580c4bfc9bd19891c2a6a4.
Its review and validator hashes must still match; old bundles are not
silently rebound. All six full-size XYZ windows, their CPU checks and the
actual paired convergence result remain unfinished. Training-data
readiness stays false.
