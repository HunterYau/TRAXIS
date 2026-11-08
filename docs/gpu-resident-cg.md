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
