# Float64 CUDA element operators

This optional backend streams the checked native H1 element stiffness and
scalar consistent-mass matrices to a CUDA device. It preserves three-component
elasticity, physical units, quadrature, mesh and coefficient orientations.
No global sparse matrix is assembled by the GPU backend. It stores local
element matrices, rather than recomputing every element integral at each CG
application as the CPU nonassembled reference does.

This is a mechanical solver candidate. No bonded PZT/electrode/adhesive model,
physical voltage calibration or accepted research waveform bank is implied.
The installed NGSolve build still has CUDA disabled: GPU application here is
implemented separately through CuPy/NVRTC, while native NGSolve supplies the
checked spaces and local finite-element matrices.

## Optional environment

The tested environment is Windows x64 / CPython 3.12, with the existing
NumPy 2.5.4 and SciPy 1.18.1 preserved. Install the core and high-order
requirements first, then the hash-pinned optional runtime:

```powershell
.\.venv\Scripts\python.exe -m pip --isolated --disable-pip-version-check install --no-deps --only-binary=:all: --require-hashes --no-cache-dir --index-url https://pypi.org/simple -r requirements-gpu-win-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

The five pinned wheels are CuPy CUDA12x 14.2.0, cuda-pathfinder 1.8.6,
CUDA runtime 12.9.79, NVRTC 12.9.86 and CCCL 12.9.27. Installation stayed in
the project virtual environment. No driver, global toolkit, compiler, PATH
or registry change was made. A tiny float64 CUDA kernel matched its exact
NumPy reference, and dependency checks reported no broken requirements.
This Windows lock does not apply to macOS or other Python wheel tags.

CuPy's official [installation guide](https://docs.cupy.dev/en/stable/install.html)
describes component-wheel discovery. Logs retain a nonfatal missing
system-CUDA-path warning; actual NVRTC compilation and execution passed.
Only basic custom kernels/copies are used, not the uninstalled optional
cuBLAS/cuFFT/cuSOLVER libraries or CuPy's SciPy-extra dependency set.

## Layout, storage and equation

For each element, native GetFE/GetTrafo/integrators compute the full vector
K_e and scalar M_e. The scalar element coefficient map is checked against
all three components of the native compound space. Signed edge/face basis
orientations are retained. A CUDA row kernel gathers coefficients, performs
float64 local products and atomically adds to the corresponding global row.
It requires hardware double atomic addition. It uses no fast-math option.

The three components share scalar consistent mass; this is storage reuse,
not lumping. Exact K/M diagonals are accumulated from the same native local
matrices. Standard operator scaling/addition fuses M + dt^2 K/4 into one
CUDA traversal and one host transfer, while retaining the original K and M
for energy checks. SciPy's checked CG/Newmark workflow remains unchanged.
Returned host vectors are independent, and shared device buffers are locked.

Before native or CUDA allocation, topology/count/vector and device-storage
screens apply. Device storage is bounded explicitly and must fit within 85%
of currently free VRAM. Default element chunks contain 128 elements; the
full-size P3 host-matrix chunk is 3.906 MiB. The outer guard still enforces
1 GiB combined RSS, 85% system RAM, at least 2 GiB available, 90% system GPU
memory and 80 C GPU temperature. CPU temperature remains unavailable here.

Fresh atomic sums may differ in low bits across executions. Verification uses
the fixed numerical tolerances; no bitwise repeatability is claimed for fresh
GPU solves. Saved array hashes bind the actual retained results.

## Small independent-reference checks

```powershell
.\.venv\Scripts\python.exe -m traxis.gpu_verification --output outputs/gpu-reference --max-seconds 180 --max-rss-mib 1024
```

The eight cases cover P1-P4 on rectangular and affine-skewed tiny blocks.
Every coefficient column of K, M and the fused Newmark operator is compared
with the native sparse reference. Diagonals, block application, returned-buffer
independence and 32-step transients are checked too. Rectangular loads/probes
are circular mechanical covectors; skewed cases use manufactured covectors.
This deliberately includes tiny sparse reference assembly for verification.

The initial eight-case run completed in 16.16 monitored worker seconds.
Worst waveform difference was 2.75e-11, maximum true residual 1.001e-11 and
postburst relative energy range 1.42e-12. All fixed 1e-10 operator, 1e-7
waveform, 1e-9 residual and 1e-8 energy criteria passed. Peak sampled combined
RSS was 427.6 MiB, GPU system memory 865/8,192 MiB and GPU temperature 53 C.
These timings are individual observations, not a controlled throughput claim.

The actual first-step driver also passed a streamed, multi-chunk small P2
comparison to sparse acceleration and direct Newmark. Eleven focused tests
cover those cases, fusion/scalar/block/transpose behavior, preflight limits,
the CPU step driver and the shared integrator. The complete native/GPU-enabled
regression passed all 247 tests without skips in 473.383 test seconds /
477.001 monitored seconds. Peak sampled combined RSS was 830.89 MiB, system
RAM 47.1%, available memory at least 8,583.19 MiB and GPU temperature 46 C.

## Actual 500 mm first-step measurement

```powershell
.\.venv\Scripts\python.exe -m examples.matrixfree_step --capacity outputs/matrixfree-fullplate-capacity-20261107 --settings configs/gpu-first-step.json --output outputs/gpu-first-step --max-seconds 180 --max-rss-mib 1024
```

The retained intact mesh, material, 3 mm-radius force disk at (20,20) mm,
100 kHz/two-cycle Hann source, 125 ns step and strict solver criteria are
unchanged from the [CPU cost probe](matrixfree-step-cost.md). GPU storage
has an explicit 3,072 MiB screen. No global sparse reference is assembled.

The 73,487 elements and 1,319,508 vector coefficients require 2,242.65 MiB
of local matrices; the conservative matrices/maps/vector screen is 2,328.79
MiB. Upload and native setup completed at 49.76 worker seconds. The first
acceleration equation converged in 242 iterations and 12.08 solve seconds,
with independently checked residual 9.53e-12. Its source-mean displacement
is 3.29373e-15 m and work/energy balance error 6.66e-16. One step completed.

Outer worker wall time was 78.80 seconds. Peak sampled combined RSS was
568.20 MiB, system RAM 45.3%, available memory at least 8,876.9 MiB, GPU
system memory 3,135/8,192 MiB and GPU temperature 63 C. No full training ran.
The CPU attempt stopped at 180 seconds without a checked step; no exact CPU
completion time or controlled speedup ratio is asserted.

Plans bind source, both drivers, retained mesh inputs, native/GPU package
versions and library hashes. Stage/upload/iteration progress, source/load and
acceleration archives and resource receipts remain under
`outputs/gpu-fullplate-first-step-20261107`. Verification data is under
`outputs/gpu-element-all-orders-20261107`, with the small direct step in
`outputs/gpu-first-step-direct-20261107`. Compiler caches/temporary directories
are placed in each command's project output directory.

This removes the observed first-step cost stop for this configuration. It
does not establish affordable full-window propagation, full-size mesh/time
accuracy, representative defect/frequency responses or calibrated PZT voltage.
Research-data readiness remains false. Longer guarded propagation and the
remaining physical checks are required before generating research banks.

## Independent CPU check of the full-size saved state

```powershell
.\.venv\Scripts\python.exe -m examples.verify_gpu_step --gpu-result outputs/gpu-fullplate-first-step-20261107 --output outputs/gpu-step-cpu-check --max-seconds 120 --max-rss-mib 1024
```

The read-only checker rebuilds the native CPU nonassembled operators and
circular load, then applies them to the saved GPU acceleration. It verifies
the frozen inputs, source/runtime and bounded float64 archives before loading.
No CPU CG solve, global sparse matrix or CUDA context is needed.

The full-size native CPU residual is 9.527939665e-12, compared with the GPU's
9.527926869e-12. The reconstructed load agrees exactly; energy and probe
differences are 4.44e-16 and 2.22e-16. CPU work/energy balance error is zero
at reported precision. The check took 36.91 monitored seconds at 404.15 MiB
combined RSS. Seven separate input checks pass, including rejection of changed
acceleration, incomplete receipt, wrong dtype/shape, object arrays and expanded
archive overflow. This checker was added after the 247-test suite and verified
separately. It checks the execution path, not an independent physical model.

Compact source/package/file identities, metrics, hashes and monitor/test
receipts are retained in `evidence/gpu-element-debug.json`. Raw CPU check and
input-check evidence are under `outputs/gpu-fullplate-cpu-residual-20261107`
and `outputs/gpu-step-input-checks-20261107`.

The optional [element preconditioner](gpu-element-preconditioning.md) adds a
separately verified SPD local inverse. Its full-size cost/memory tradeoff and
subsequent 250-test integrated regression are recorded separately; the diagonal
baseline described above remains available.
