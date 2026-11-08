# Element preconditioning for the CUDA reference

The optional element additive Schwarz preconditioner targets the measured
iteration cost of the full-size P3 mechanical system. It changes the CG search
directions, while preserving the full consistent-mass equation, source, mesh,
time step, residual target and mechanical energy checks.

For the fixed effective matrix A = M + dt^2 K/4, each element contributes
A_e = M_e + dt^2 K_e/4. Native local matrices are inverted in bounded CPU
chunks using Cholesky factors. Each inverse must satisfy a 1e-9 normalized
identity check. The inverse is stored in float64 on the GPU.

Let R_e extract the element's signed native coefficients, and D be the diagonal
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
