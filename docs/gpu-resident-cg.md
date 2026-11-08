# GPU-resident conjugate gradients

The optional `gpu_cg` solver keeps CG coefficient vectors and reductions on
the CUDA device. Scalar reductions synchronize with the host, but iteration
vectors are not copied back for SciPy/NumPy arithmetic. The original native
element matrices, full consistent mass, optional Schwarz preconditioner and
strict final equation check remain in use. The default remains `scipy_cg`.

`traxis/gpu_cg.py` accepts checked GPU element operators and finite real host
vectors. It uploads the right-hand side and initial guess once, plus the
diagonal when using Jacobi. Each iteration applies the original operator,
checks positive curvature and preconditioned work, and updates float64 device
vectors. A successful solve must reach the recurrence target and pass a fresh
true residual. Nonconvergence is rejected even if the looser final acceptance
threshold happens to pass. The result is copied to host once.

The retained statistics distinguish vector uploads/downloads from scalar
readbacks. Progress labels recurrence residuals separately from the final
true residual. Zero right-hand sides return the unique zero solution of the
declared SPD equation; caller guesses are preserved. Invalid inputs and vector
workspace limits are checked before device allocation. Device storage screens
now reserve sixteen coefficient vectors, including operator scratch and CG
workspace. The outer RAM/VRAM/temperature/time guard remains mandatory.

The element kernels also avoid integer division inside their inner component
loops. Local coefficient order and summation order remain the same; every
operator/preconditioner column is rechecked against native references. No
fast-math option, mass lumping, reduced plate or relaxed tolerance is used.

```powershell
.\.venv\Scripts\python.exe -m traxis.gpu_verification --preconditioner element_schwarz --solver gpu_cg --output outputs/resident-reference --max-seconds 150
.\.venv\Scripts\python.exe -m examples.matrixfree_step --capacity outputs/matrixfree-fullplate-capacity-20261107 --settings configs/gpu-resident-first-step.json --output outputs/resident-first-step --max-seconds 180
.\.venv\Scripts\python.exe -m examples.verify_gpu_step --gpu-result outputs/resident-first-step --output outputs/resident-cpu-check --max-seconds 120
```

Use fresh output directories and retain their plans, source snapshots, arrays
and monitor receipts. Historical states remain bound to their original code;
source checks must not be bypassed after changing the solver implementation.

The existing host-state Newmark integrator can select `solver='gpu_cg'` for
its linear solves. Its Newmark state updates, source/probe handling and energy
checks still run on the CPU. This does not yet provide a completely device-
resident propagation loop or establish full-window cost. The small verification
compares every K/M/effective/preconditioner column and 32-step P1-P4 rectangular/
affine transients with the native sparse reference. Separate tests cover warm
starts, exact zero forcing, nonconvergence and fixed vector-transfer counts.

Full-size propagation, mesh/time refinement, representative recess/frequency
variation, coupled/calibrated PZT voltage and physical agreement remain open.
No research bank or full training cycle is authorized by these software checks.

## Retained measurements, November 7

All eight P1-P4 rectangular/affine GPU-resident cases pass every-column and
32-step direct checks. Worst waveform difference is 2.44e-11, true residual
9.994e-12 and postburst relative energy range 6.85e-13. The retained run took
16.82 monitored seconds. Seventeen focused tests passed in 82.964 test seconds /
84.146 monitored seconds, including the real small-step driver. The native
contract probe was then isolated to release its CUDA context between test
groups, and that targeted guarded check passed too.

The full-size P3 mesh/source/material/125 ns increment are unchanged from
the [host-controlled preconditioner measurement](gpu-element-preconditioning.md).
Device CG converges in 70 iterations / 5.326 solve seconds versus the earlier
70 / 7.032. Total monitored time is 93.573 seconds, versus 95.169 previously.
The kernels' inner loops also changed; these separate observations do not
isolate an individual optimization or establish a controlled speedup ratio.

Only two coefficient vectors were uploaded and one downloaded during CG;
213 scalar reductions were read back. The checked residual is 7.438959e-12,
source mean displacement 3.293734e-15 m and work/energy error 2.22e-16.
Peak sampled combined RSS is 599.82 MiB, system RAM 48.2%, available RAM at
least 8,403.89 MiB, GPU memory 5,352/8,192 MiB and temperature 66 C. CPU
temperature remains unavailable. No resource limit was raised.

A separate native CPU action check passes with residual 7.438951e-12. The
reconstructed load agrees exactly; CPU/GPU energy agrees at reported precision
and probe difference is 2.22e-16. The check took 36.64 monitored seconds at
384.31 MiB combined RSS, without CPU CG or global sparse assembly.

Raw artifacts are under `outputs/gpu-resident-reference-20261107`,
`outputs/gpu-resident-fullplate-step-20261107`,
`outputs/gpu-resident-cpu-check-20261107`,
`outputs/gpu-resident-focused-20261107` and
`outputs/gpu-resident-isolation-check-20261107`.

All 253 integrated regression tests pass without skips with real Elmer, Gmsh,
NGSolve and GPU checks enabled. Test time is 525.159 seconds; monitored time
is 528.187 seconds. Peak sampled combined RSS is 875.42 MiB, system RAM 49.9%,
available RAM at least 8,135.02 MiB, GPU memory 978 MiB and temperature 43 C.
CPU temperature remains unavailable. No full training ran.

The saved acceleration differs from the earlier host-controlled Schwarz
result by 2.04e-15 relative L2. Retained mesh/physical settings were checked
equal, while each state's source identity remains separate. Compact plans,
source/package identities, array hashes, counters and test/resource receipts
are retained in `evidence/gpu-resident-debug.json`, fingerprint
`a9729e8fd57620682d408828b596a4e8f890ad0e99294a01466bebd7b872b878`.
Complete regression logs are under
`outputs/gpu-resident-native-regression-20261107`.
