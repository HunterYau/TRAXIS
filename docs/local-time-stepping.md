# Broadband with local time stepping

Branch: `research/broadband-local-time-stepping`, based directly on
`research/broadband-comparison` at `581f273`. The MOR branch and its working
files remain separate. This worktree is `.worktrees/local-time-stepping`.

## Method and scope

Implementation: Grote and Mitkova, *Explicit local time-stepping methods for
time-dependent wave propagation*, section 3.1, LTS-LF2(p) algorithm:
https://arxiv.org/pdf/1205.0654 . Original method: Diaz and Grote (2009),
https://doi.org/10.1137/070709414 . This is the published second-order,
lossless leapfrog recurrence. It does not use independent regional Verlet
updates or frozen interface tractions.

`traxis/local_time.py` implements the recurrence. `traxis/gpu_local_time.py`
applies it using the actual tensor plate and resolved PZT body operators on
CUDA in float64. A coarse step H contains p fine steps H/p. P selects all
resolved PZT mass blocks and the drive support, plus optional spatial boxes
or an edge strip. A plain straight edge does not automatically require fine
steps; selecting one is an explicit experiment choice. No mesh is coarsened.
Defects are not added to the geometry by a fine selection box.

The action AP = M^-1 K P evaluates only tensor elements touching the fine
input support, plus exact PZT stiffness blocks. It retains EVERY output row
including neighboring coarse/interface rows. Whole coupled consistent-mass
blocks remain fine so P commutes with the mass transform. No PZT mass lumping,
material changes, damping, pole correction, time alignment or gain fit is added.
Nonzero damping, finite-resistance electrical ports and unsupported operator
types are explicitly rejected. The original broadband branch is lossless
with condensed ideal-open receiver ports and is supported as-is.

## Stability, phase and speed

For now, H must be at most 0.8 times the existing full-system stable dt.
The code does not assume that selecting the PZTs removes their global CFL
restriction. The original LF-LTS method can have stability restrictions;
relaxed CFL values need a separate certification study. The initial study
therefore refines PZT timing relative to the original coarse step, comparing
against uniform fine stepping. It is not yet a demonstrated acceleration
of the already validated original broadband run.

Local stepping still has phase error. Small tests establish order and
correctness, not full-plate A0/S0 phase agreement. Use full signed voltage,
sum/difference traces, arrival and envelope metrics without phase fitting.
Do not describe this branch as production validated until those checks pass.

Broadband needs p phase kernels because the update is periodic on the fine
input grid. Kernel r is a unit source at fine index r+1. Reconstruct by
convolving each kernel with its matching phase of the input and summing.
The original single Verlet kernel cannot be reused. CPU tests compare this
directly to the same LTS solver with a varying source, to roundoff. Multiple
kernels add setup cost and can erase the gain for a single frequency batch.
Report total kernel time, all reconstruction time, and all direct times.
No speedup or runtime estimate is claimed before measurement.

## Scheduled comparison (not started)

Run from this worktree/branch using a working Python 3.12 environment and
the existing project dataset/configuration files (ignored data files are
not copied automatically into a Git worktree):

```powershell
python -m examples.pzt_local_time_study --prepare --root outputs/simulation-runs/lts-003 --substeps 2
python -m examples.pzt_local_time_study --run --root outputs/simulation-runs/lts-003
```

Preparation only writes a plan. Execution is a separate explicit action.
The local `lts-003` plan has already been prepared; use only `--run` for
that existing plan. Choose a fresh directory when preparing another plan.
The default schedule contains the original broadband control kernel, two phase kernels and uniform-fine/LTS direct
pairs at 50, 100, 150, 200, 250 kHz. H is 350 us / 65536 = 5.340576171875 ns;
for p=2, h is 2.6702880859375 ns. Traces are compared on identical H nodes;
fine input nodes are retained for reconstruction. No interpolation aligns
the compared output traces. Use `--frequencies 100000` for a smaller first
comparison, and `--fine-box xmin xmax ymin ymax zmin zmax` or
`--edge-width-m value` to explicitly enlarge the fine region.

Each task writes progress and a receipt in its own directory. A failed or
interrupted task retains its saved files, is marked incomplete, and is rerun
from t=0 on restart; completed SHA256-verified tasks are reused. In-memory
states are not checkpointed. The run is locked against a duplicate scheduler.
SIGTERM is converted to an interruption so the incomplete task is marked.
Every run records the LTS-specific M-weighted discrete energy. After source
excitation ends, relative drift greater than 1e-8 rejects that task. Original
Verlet controls use their own invariant; these are different discrete energies.
`performance.json` compares original versus LTS broadband kernel and
reconstruction totals, with the shared operator build reported separately.
`all-results.txt` stores the collective comparison and task timing record.
No pod creation, start, pause or stop code is present.

The full study defaults to an explicit 20000 MiB cloud device budget. It checks
actual free VRAM before building the full plate, and fails on insufficient
hardware. This is not an allocation promise or a local GPU configuration.
The integrator screens peak vector workspace as 32 float64 full-size vectors
plus traces, masks and element indices, in addition to operator storage.
The configured total device budget and actual free VRAM are checked before
allocation. The order-5 baseline has about 28 million DOFs; the vector allowance
alone is about 6.7 GiB (the pre-build operator-plus-vector screen is about 10 GiB);
PZT matrices and the plate operator are additional. This is not a verified
fit on the local 8 GB card. Check host and device memory before any full run;
small CUDA checks here allocate only tiny plates. No full simulation was run.

## Checks

`tests/test_local_time.py`: independent modal temporal convergence, exact
ratio-one leapfrog reduction, coupled-mass symmetry and long discrete-energy
conservation, multirate impulse superposition at p=1/2/4, partition closure,
element support, invalid arguments.

`tests/test_gpu_local_time.py`: real CUDA selected-element action against the
independent CPU tensor operator, retained interface rows, ratio-one agreement
with existing GPU Verlet, exact resolved-body mass action and mass-block
closure, rejection of unsupported damping and unstable coarse steps.

The additional audit and its checked results are in `local-time-stepping-audit.md`.
These checks authorize further validation, not a claim of full-plate phase,
mesh convergence, measured-data accuracy, or speedup.
