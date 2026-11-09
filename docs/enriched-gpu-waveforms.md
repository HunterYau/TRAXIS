# Factorized enriched GPU elasticity

The 15-node positive-mass candidate now has a float64 GPU stiffness action and
device-resident velocity Verlet. It retains the vector elastic weak form and
all enriched coefficients. The shared reference-gradient Gram factor has rank
13; this removes redundant gradient data, not displacement modes. Each element
stores geometry and fifteen scalar indices rather than a dense 45-by-45 matrix.

`gpu_enriched.py` checks every native element's nodal permutation, shared
entities, component ranges, positive mass and coordinates. Its maximum local
mass-normalized Gershgorin bound conservatively limits the assembled explicit
step. The integrator applies an additional 0.95 safety factor. Circular
footprints use cubic surface traces despite the enriched space advertising
order two. A quadratic-only trace request is rejected by the retained control.

`gpu_enriched_verlet.py` keeps states, stiffness actions, mass division and
probe accumulation on the GPU. No coefficient vector moves to the host per
step. Scalar energy/resource observations still synchronize. Probe compression
drops no nonzero coefficients. Requested intervals are never shortened to fit
a guard. Inputs, sample storage, vector workspace and free VRAM are screened.

## Retained small references

`outputs/enriched-gpu-corrected-reference-20261108` checks every column on two
six-tetrahedron controls with 153 vector coefficients each. Operator discrepancies
are about 1.2e-15; positive masses and native/topology maps agree. GPU/CPU Verlet
traces agree at 64/128/256 steps over each fixed manufactured-mode interval.
The rectangular driven disk experiment covers 20 microseconds at 200 kHz,
including reciprocity and postburst modified-energy checks. It is not a
full-plate acquisition. Only the rectangular case runs the driven patch checks.

The earlier `enriched-gpu-reference-20261108` report overwrote its stiffness
error variable with the final unforced waveform error. Its saved raw matrices
independently pass the intended comparison, but its mislabeled field and gate
must not be used as that proof. The corrected driver keeps distinct variables;
the test recomputes the reported stiffness norm from the saved matrices.
Unperformed driven checks now use null values and an explicit status flag.

All 17 selected enriched GPU/element/circular-patch tests pass without skips in
23.454 test seconds. The final standalone reference also passes and retains
plans, source/runtime/package bindings, arrays and monitor receipts. The last
complete integrated suite remains the earlier 253-test result; overlapping
selected suites are not added to that total.

## Full-size operator measurement

`examples.enriched_gpu_wave` binds the retained capacity mesh, source, driver,
native/GPU packages, explicit settings and acceptance criteria. Operator mode
does not propagate; wave mode must finish its entire declared window. Both
compare a nonaffine physical-coordinate GPU field action with independent
streamed exact local CPU matrices. The native-order positive mass is checked
against all intact-box moments through degree two. Circular patches check
their area and represented quadratic moments before propagation.

The actual unchanged 500 x 500 x 3 mm mesh has 73,487 tetrahedra and 1,173,621
vector enriched coefficients. Every element's native/topology map passes.
The independent full-size action discrepancy is 4.44948e-15 relative L2.
The conservative explicit limit is 19.8610 ns. At CFL fraction 0.8, the declared
500-microsecond window requires 31,469 steps of 15.8887 ns. Twenty repeated
device actions average 17.8820 ms wall / 17.8489 ms device time, versus 29.5028
ms for the host-facing action. The independent CPU action takes 2.59161 seconds.
No full-window throughput follows from this isolated action measurement.

Raw evidence is in `outputs/enriched-gpu-fullplate-operator-20261108`, which
completed in 48.3122 monitored seconds. The two preflight tests pass and keep
the observation window, stability/accuracy criteria, geometry and GPU limits
explicit. Settings are in `configs/enriched-gpu-wave.json`.

## Complete 500-microsecond mechanical window

The same full-size mesh completes all 31,469 steps in 216.077 propagation
seconds / 266.092 monitored seconds. It uses the declared 100 kHz, two-cycle
Hann force scale of 1 N on the normal 3 mm disk at (20,20) mm. Free finite
edges remain present. Normal-displacement probes at (250,250) and (480,480) mm
retain all 31,470 samples through exactly 500 microseconds. Peak magnitudes
are 0.500490 and 0.847393 nm. These are mechanical displacement, not voltage.

Postburst modified-energy relative range is 6.46940e-16, below the fixed 1e-8
screen. The run performs 31,470 stiffness actions, transfers no coefficient
state per step and downloads two final state vectors. Its explicit workspace
screen is 143.79 MiB. Measured peak combined RSS is 644.25 MiB, system RAM 57.2%,
available RAM at least 6,954.33 MiB, system-wide GPU memory 3,503/8,192 MiB and
GPU temperature 77 C. Guards remain 1 GiB combined RSS, 85% system RAM, 2 GiB
available RAM, 90% VRAM and 80 C GPU temperature. CPU temperature is unavailable.

The warm sustained propagation cost differs from the short isolated action
forecast; the full measured interval is the throughput evidence. A separate
read-only CPU audit checks file hashes, sample times, receiver covectors and
recomputes positive mass and final Ku from exact streamed local stiffness.
Mass agrees exactly, final physical/modified energies agree to 3.08e-14
relative and final probes to 3.96e-16 relative L2. This is not a CPU replay of
the entire history. Raw evidence is under
`outputs/enriched-gpu-fullplate-window-20261108` and
`outputs/enriched-gpu-evidence-tools-20261108`; the latter contains PNG/PDF
trace plots and the independent CPU-check receipt.

## Independent dispersion study

`examples.enriched_dispersion` compares enriched periodic-cell eigenvalues
with Rayleigh-Lamb roots at 100/200 kHz in 1.5/3 mm stock. Only opposite lateral
faces are tied; enriched interior coefficients remain free. The component-major
x/z sector excludes SH motion. This benchmark's periodic faces do not replace
the finite physical edges in the full waveform. Positive nodal mass is used,
with exact enriched stiffness, retaining its intentional difference from the
ordinary consistent-mass discretization.

The 72 cases independently vary 4/8/16 cells per wavelength and 1/2/4 thickness
cells for both A0/S0. All eight finest cases meet the fixed 2% frequency screen;
the largest error is 0.0250435%. Coarse A0 errors reach 3.66794%. At 100 kHz in
3 mm stock, the 4-by-1 A0 error is 2.62969%, versus 0.0194325% for 16-by-4.
With one thickness cell, lateral refinement from 8 to 16 changes the A0 error
from 1.25245% to 1.39808%: lateral refinement alone is not a convergence proof.
These structured-cell errors do not estimate the error of the full unstructured
mesh. Full-size space/thickness/time studies remain necessary.

The study completes in 4.481 monitored seconds. Two independent projection
tests check phase-one/Bloch face ties, free interior coefficients, rank and
invalid/unmatched geometry rejection. The evidence audit verifies all 72
archive hashes, independently recomputes their generalized eigenvalues and
rechecks their analytical wavenumbers. Raw results are under
`outputs/enriched-lamb-dispersion-20261108`. Compact identities, metrics and
resource summaries are in `evidence/enriched-gpu-wave-debug.json`.

```powershell
.\.venv\Scripts\python.exe -m examples.enriched_gpu_wave --capacity outputs/matrixfree-fullplate-capacity-20261107 --mode wave --output outputs/enriched-window --max-seconds 900
.\.venv\Scripts\python.exe -m examples.enriched_dispersion --output outputs/enriched-dispersion
```

Use fresh output directories and the retained capacity inputs. The settings
file keeps the window and criteria explicit. Operator mode runs no time steps.

## Remaining acceptance

Spatial/thickness/time convergence,
Lamb-wave dispersion, representative defects/frequencies, coupled or calibrated
PZT voltage and physical agreement remain required. No research bank is
accepted and no full training ran. `ready_for_training_data` remains false.
