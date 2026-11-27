# Precomputed spectral PZT GPU execution

The Q2 6-ring and 8-ring study now has a precomputed option. It uses the existing
dense PZT CUDA kernel and the existing optimized LTS executor, including CUDA
graphs, reusable buffers and fused vector arithmetic. The sparse factor action
backend remains available as a reference and memory-saving fallback.

The change is confined to operator representation and study dispatch. The Q2
displacement and electric potential, Gauss order 6, interface map, material
properties, full consistent mass, source and exact time grid are retained.
The original LTS sources and optimized executor are unchanged.

Each PZT's transformed stiffness includes the same electrical Schur complement
and open-port rank-one term. Its full mass matrix and inverse are precomputed
from the same sparse mass LU factor. Bounded column batches avoid creating a
full component inverse or large identity matrix. Patches upload one at a time;
the GPU retains their dense matrices, while their host dense arrays are released.
Three random action/residual checks run for every patch before upload.

## Prepared schedule

`outputs/simulation-runs/spectral-q2-precomputed-001` contains the new frozen
schedule and imported historical reference metadata. The existing sparse run
folder is preserved. The new schedule computes a Q2 six-ring reference, followed
by eight rings, and reconstructs the six 50-300 kHz signals with all four channels.
The existing study comparison, plotting, failure and completed-kernel receipts
are retained. There are no completed spectral results to reuse locally.

| Rings / sides / layers | DOFs | GPU peak forecast | Host peak forecast |
| --- | ---: | ---: | ---: |
| 6 / 64 / 1 | 28,217,268 | 34.03 GiB | 10.85 GiB |
| 8 / 64 / 1 | 28,226,484 | 42.70 GiB | 13.88 GiB |

These are planning estimates, not measured peaks or confirmation of a particular
cloud allocation. Actual available host and GPU memory are checked before builds.
The full study does not fit the local 8 GiB GPU. No full simulations were launched.
Matrix setup takes additional time; full end-to-end speed has not been measured.

The requested coarse step is exactly 10.68115234375 ns; the fine step is exactly
5.340576171875 ns. The inherited study has stability screening disabled and no
half-step temporal qualification; this execution change does not validate that
choice or select another time grid.

Recreate the schedule using:

```bash
python -m examples.pzt_mesh_study --prepare --pzt-element spectral \
  --pzt-storage precomputed --execution-backend optimized --skip-stability-checks \
  --root outputs/simulation-runs/spectral-q2-precomputed-001 \
  --baseline outputs/simulation-runs/spectral-q2-convergence-001/baseline \
  --assembly-sources outputs/simulation-runs/spectral-q2-convergence-001/baseline-assembly-sources
```

Preparation requires a new, nonexistent destination. The source files and new
prepared folder must be transferred before cloud execution. The old
`spectral-q2-transfer-001.tar.gz` archive predates this change and does not contain
the restored fast path. After transferring current files, preflight and launch:

```bash
bash tools/runpod/spectral_q2_setup.sh
bash tools/runpod/spectral_q2_run.sh
```

The launcher defaults to the new precomputed schedule. `TRAXIS_STUDY_ROOT` can
select another prepared schedule. `--retry-failed` is forwarded explicitly when
needed. Setup includes CUDA 12 cuBLAS/cuSOLVER dependencies and an actual tiny
GPU sparse solve, so missing runtime libraries are detected during setup.
Neither script changes pod lifecycle state.

## Verification

- Independent dense dielectric and mass calculations: both plate faces,
  voltage/open ports, reversed poling, input capacitance, and remainder batches.
- Tiny three-PZT real CUDA check on the RTX 3060 Ti: 1,431 DOFs, 65 steps, graph
  replay enabled and disabled, awkward graph and energy sampling boundaries.
- Largest relative voltage waveform difference: 1.21e-14. Largest envelope
  difference: 1.23e-14. Energy diagnostic difference: 1.76e-15.
- 21 relevant CPU tests: 20 passed, one unavailable archived-fixture skip.
  The new schedule's separate preflight verifies the actual imported baseline
  hashes, assembly compatibility, fixed physical parameters and frozen sources.
- Protected original solver, optimized executor, element and sparse operator
  SHA-256 hashes and Git diff are checked under
  `outputs/spectral-precomputed-check-001`.

The GPU check saves `gpu-check.json`. Its short timings include differing first
use and graph capture costs and must not be extrapolated to a full plate speedup.
No approximation, precision reduction, mass lumping, waveform alignment or
amplitude normalization is introduced by this backend.
