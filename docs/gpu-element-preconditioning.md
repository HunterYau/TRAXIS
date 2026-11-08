# Element preconditioning for the CUDA reference

The optional element additive Schwarz preconditioner targets the measured
iteration cost of the full-size P3 mechanical system. It changes the CG search
directions, while preserving the full consistent-mass equation, source, mesh,
time step, residual target and mechanical energy checks.

For the fixed effective matrix A = M + dt^2 K/4, each element contributes
A_e = M_e + dt^2 K_e/4. Native local matrices are inverted in bounded CPU
chunks using Cholesky factors. Each inverse must satisfy a 1e-9 normalized
identity check. The inverse is stored in float64 on the GPU.

Let R_e extract the element's coefficients in native orientation, and D be the diagonal
reciprocal of each coefficient's element multiplicity. The applied inverse is

    B = D (sum_e R_e.T inverse(A_e) R_e) D.

Both weights are required for symmetry. Positive local consistent mass makes
each effective element matrix positive definite; full coefficient coverage
and positive weights make B positive definite. CUDA application uses the same
checked coefficient map and float64 atomic scatter as the elastic operators.
Fresh summation can vary in low bits. All accepted CG states still require a
fresh residual computed from the original equation.

The extra local inverses require an explicit larger device-memory budget.
Storage screens include the inverses, weights and bounded factorization
workspace before native/device allocation; the 85% free-VRAM screen and outer
resource guard remain active. This preconditioner is optional. The diagonal
baseline remains the default and can be selected for comparison.

```powershell
.\.venv\Scripts\python.exe -m traxis.gpu_verification --preconditioner element_schwarz --output outputs/schwarz-reference --max-seconds 150
.\.venv\Scripts\python.exe -m examples.matrixfree_step --capacity outputs/matrixfree-fullplate-capacity-20261107 --settings configs/gpu-schwarz-first-step.json --output outputs/schwarz-first-step --max-seconds 180
.\.venv\Scripts\python.exe -m examples.verify_gpu_step --gpu-result outputs/schwarz-first-step --output outputs/schwarz-cpu-check --max-seconds 120
```

Use fresh output directories. The saved plans bind current source, runtime,
retained mesh and settings. Historical results must be checked using their
recorded source, rather than bypassing identity checks after editing code.

The small verification assembles an independent CPU Schwarz reference using
local dense solves. It compares every GPU preconditioner column, checks full
matrix symmetry and Cholesky positivity, and compares 32-step transients with
direct Newmark for rectangular and affine P1-P4 cases. The existing every-column
K/M/effective-operator, diagonal and buffer tests remain enabled. Full-size
measurements and independent CPU residual checks are recorded separately.

No first-step or small-reference success establishes full-window propagation,
full-size mesh/time convergence, defect/frequency variation, bonded PZT voltage
or agreement with physical measurements. Research-data readiness remains false.

## Retained measurements, November 7

All eight rectangular/affine P1-P4 references pass. Worst preconditioner
matrix difference is 1.34e-13 and relative skew is below 5.04e-18. Every
represented inverse has a positive Cholesky factor. Worst 32-step waveform
difference is 2.04e-11, with the original residual/energy criteria unchanged.
The retained reference took 13.62 monitored seconds. Fourteen focused tests
pass without skips in 56.820 test seconds / 57.966 monitored seconds.

The retained 500 x 500 x 3 mm P3 mesh, physical source and 125 ns time step
are unchanged from the diagonal-preconditioned GPU measurement.

| Observation | Earlier diagonal run | Element Schwarz run |
| --- | ---: | ---: |
| CG iterations | 242 | 70 |
| Solve seconds | 12.080 | 7.032 |
| Total monitored seconds | 78.801 | 95.169 |
| Peak sampled combined RSS, MiB | 568.20 | 586.98 |
| System GPU memory, MiB | 3,135 | 5,264 |
| Maximum system GPU temperature, C | 63 | 64 |
| Checked relative residual | 9.53e-12 | 7.44e-12 |

The solve uses fewer iterations and less time, while the added factorization
increases setup and total time. These are separate observed runs, not a
controlled throughput benchmark or proof of affordable full-window propagation.
Setup is intended to be reused within a fixed-mesh, fixed-time-step transient;
that longer workflow has not yet been measured.

Full-size storage screens are 4,350.52 MiB device and 25.0 MiB host chunk,
including 2,021.74 MiB extra inverse/weight storage. The maximum local inverse
identity error is 9.94e-13; coefficient multiplicities range from 1 to 22.
Native setup/upload completed at 71.40 worker seconds. System RAM reached
48.6%, with at least 8,339.71 MiB available; CPU temperature was unavailable.

A separate native CPU action check of the saved acceleration passes with
residual 7.439070895e-12 versus GPU 7.439083102e-12. The reconstructed load
agrees exactly. CPU/GPU energy and probe differences are 1.11e-16 and 2.22e-16.
The check took 37.35 monitored seconds at 384.61 MiB combined RSS. It used
no CPU CG solve or global sparse matrix. No full training ran.

Raw evidence is retained under `outputs/gpu-schwarz-reference-20261107`,
`outputs/gpu-schwarz-fullplate-step-20261107`,
`outputs/gpu-schwarz-cpu-check-20261107` and
`outputs/gpu-schwarz-focused-20261107`.

All 250 integrated regression tests pass without skips with the real Elmer,
Gmsh, NGSolve and GPU checks enabled. Test time is 495.937 seconds; monitored
time is 498.918 seconds. Peak sampled combined RSS is 894.28 MiB, system RAM
49.4%, available RAM at least 8,205.86 MiB, system GPU memory 976 MiB and GPU
temperature 42 C. CPU temperature remains unavailable. No full training ran.

The saved full-size acceleration differs from the earlier diagonal GPU state
by 7.65e-11 relative L2. Their retained mesh/physics settings agree; the earlier
state remains bound to its original source. The native CPU residual check is
the independent current-path check, rather than relying on historical arrays.

Compact identities, plan/report hashes, metrics and test/resource receipts
are in `evidence/gpu-schwarz-debug.json`, fingerprint
`d1cd1a20c4079e2035d8f3c7eae90081b8413b5db4e74490842bb06f065600d1`.
Full-suite logs and monitor receipts are under
`outputs/gpu-schwarz-native-regression-20261107`.

An optional [device-resident CG path](gpu-resident-cg.md) subsequently reduces
the observed full-size solve to 5.33 seconds while retaining 70 iterations.
Its separate native CPU check, revised kernel indexing, transfer counters and
253-test regression are recorded with that implementation.
