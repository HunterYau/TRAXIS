# Native matrix-free elastic reference

This candidate applies the existing three-component H1 elastic stiffness and
consistent mass through NGSolve's nonassembled forms. It preserves the same
material, coefficient basis, quadrature and natural/free boundaries used by
the checked sparse reference. It is a solver-scaling investigation; no
full-size wave, PZT voltage or training-data acceptance is implied.

## Equations and solver

The time integrator still solves `M u'' + K u = f` with average-acceleration
Newmark (`beta=1/4`, `gamma=1/2`). Each acceleration solve applies
`A = M + dt^2 K/4` through linear operators. CG uses a positive diagonal
preconditioner and an explicit iteration budget. Its declared target and an
independently recomputed residual must both pass. A nonzero CG status or
nonfinite result raises an error; it cannot silently publish traces.

The native preconditioner uses the exact diagonal of A. Element-sized scalar
mass and component stiffness diagonals are accumulated without a global
sparse matrix. For a scalar basis function phi in component a, the stiffness
diagonal integrand is `mu |grad(phi)|^2 + (lambda+mu) (d_a phi)^2`.
The full vector elastic operator, including component coupling, remains in
every equation. The full consistent mass remains in propagation and energy;
using a diagonal preconditioner does not replace it with lumped mass.

The initial acceleration uses the mass operator and its diagonal. Warm starts
are copied so CG cannot alter the preceding acceleration used in the velocity
update. Energy diagnostics use the original K/M operator actions; the verifier
checks postburst energy retention.
The operator wrapper owns its native buffers, returns independent arrays and
serializes access to those buffers. Native work uses one CPU thread.

## Verification scope

```powershell
.\.venv\Scripts\python.exe -m traxis.matrixfree_verification --output outputs/matrixfree-check --max-seconds 300 --max-rss-mib 1024
```

Use a fresh output directory. The verifier freezes source/runtime identities,
two tiny rectangular/affine-skewed meshes, P1–P4 degrees and fixed tolerances.
It compares every coefficient column against the sparse operator, exact mass
and stiffness diagonals, block application and retained output buffers. It
then compares 32-step direct/iterative transients with identical inputs and
checks true residuals and postburst energy. The rectangular block uses circular
mechanical footprints; the affine block uses explicitly manufactured
coefficient covectors. No claim of resolved continuum plate waves follows.

The verifier intentionally constructs the sparse reference on these tiny
systems. Its process memory includes that comparison work. A separate
full-size matrix-free cost measurement is needed before interpreting memory
or performance for the research plate. No global sparse matrix or direct
factorization is constructed by the candidate operator/integrator itself.

Pure regression tests compare the integrator with direct Newmark and an
independent adaptive ODE solver, on a common observation grid. They check
second-order time refinement, mixed rigid/elastic energy conservation,
nonconvergence, false solver-success statuses, complex inputs and workspace
limits. Topology and vector screens precede native allocation; an outer
process-tree guard is still required for larger native kernels.

## Recovery and measured small checks

Each completed case has a plan-bound receipt and a checked comparison archive.
Use a fresh output directory and `--cache-dir outputs/matrixfree-check/native`
to reuse it under unchanged source/runtime/settings. Recovery verifies the
complete case schema, all coefficient counts, archive expansion bounds, exact
time grids, iteration history and metrics recomputed from saved arrays.
Incomplete or altered evidence fails. Paths use forward slashes on every OS.

The first eight-case attempt stopped at P4's 500-iteration CG limit with a
2.45525e-11 residual. Its target was 1e-11, so the looser 1e-9 independent
residual limit did not override the failed CG status. Exact effective-operator
diagonals improved convergence. A later attempt hit its 180-second wall limit;
both failed attempts remain in outputs.

The compiled-form check completed all eight tiny cases in 282.62 seconds,
before subsequent recovery-validation changes. Maximum stiffness/mass errors
were 2.41e-16/2.30e-16, waveform difference 5.22e-11, true residual 1.00e-11
and postburst energy range 1.48e-12. P4 required up to 475 iterations. Sampled
combined RSS was 281.6 MiB, system RAM 54.3%, GPU memory 1,524/8,192 MiB and
GPU temperature 75 C. All calculation used CPU; CPU temperature was unavailable.
This includes tiny sparse comparison operators and is not a full-size benchmark.

After recovery hardening, all 239 tests pass without skips with real native
libraries enabled. The current-source eight-case comparison took 287.25 seconds;
checked recovery reused every case with zero new computations and identical
archive hashes. The complete regression took 475.49 monitored seconds, with
776.95 MiB sampled combined RSS, 56.8% system RAM, at least 7,015 MiB available,
GPU memory 1,536/8,192 MiB and GPU temperature 77 C. No GPU computation or full
training ran; CPU temperature was unavailable.

All eight current-source reports, receipts, arrays and source snapshot were
retained from the native test before temporary-directory cleanup, without
launching another solver run. Compact source/artifact hashes, numerical metrics,
resource summaries and the earlier failures are in
[evidence/matrixfree-debug.json](evidence/matrixfree-debug.json).

## Full-size capacity measurement

```powershell
.\.venv\Scripts\python.exe -m examples.matrixfree_capacity --settings configs/matrixfree-fullplate-capacity.json --output outputs/matrixfree-fullplate-capacity --max-seconds 180 --max-rss-mib 1024
```

This measures an actual 500 x 500 x 3 mm intact mesh, P3 native space, streamed
diagonals and two deterministic K/M applications. It saves the mesh, source
snapshot, runtime identities, action hashes, stage progress and resource report.
Checks include positive quadratic forms, work reciprocity, uniform-translation
mass and a rigid translation's stiffness null force. It constructs no global
sparse matrix and performs zero time steps or training updates.

The CAD call explicitly selects a geometry-only storage screen; existing
assembled callers retain their P1 assembly estimate. Neither screen predicts
all native allocation costs, so the outer 1 GiB, 85% system RAM, 2 GiB available
RAM and temperature guards remain active. A capacity pass is separate from
full-window propagation, mesh/time accuracy, PZT voltage and physical agreement.

The first full-size run completed in 64.15 monitored seconds on November 7.
The 5 mm intact mesh has 24,579 vertices and 73,487 tetrahedra; P3 has
439,836 scalar / 1,319,508 vector coefficients. Box-volume relative error
was 1.11e-16, translation-mass error 1.25e-14, normalized rigid force
8.88e-15 and work-reciprocity errors below 2.18e-17. The two stiffness
actions took 6.55/6.46 seconds and mass actions 2.71/2.72 seconds.

Sampled combined RSS reached 482.6 MiB, system RAM 53.9%, available RAM
stayed above 7,476 MiB, and GPU system observations reached 1,410/8,192 MiB
and 77 C. No GPU computation ran; CPU temperature was unavailable. Mesh,
progress, action hashes and receipts are in
`outputs/matrixfree-fullplate-capacity-20261107`. The operator fits this
guard, but these action costs do not establish affordable transient solves.
The conservative 8,073.5 MiB P3 assembly screen estimates a different storage
route; it is not a measured sparse allocation or measured memory saving.

## Sources

NGSolve documents nonassembled form applications in its
[operator-application tutorial](https://docu.ngsolve.org/nightly/i-tutorials/unit-3.5.1-dgapply/dgapply-scalar.html)
and [matrix-free examples](https://docu.ngsolve.org/latest/i-tutorials/unit-2.11-matrixfree/matrixfree.html).
These supply API guidance; this implementation retains the project's vector
elastic weak form and checks it against the existing independent-reference
work. The installed native build and behavior are recorded in each run.
