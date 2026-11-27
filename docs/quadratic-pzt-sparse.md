# Sparse quadratic PZT storage investigation

Implemented separately on `research/pzt-quadratic-tetrahedra`. The dense
implementation remains the default/reference. The new modules are
`traxis/quadratic_pzt_sparse.py`, `traxis/gpu_quadratic_pzt_sparse.py`, and
the construction hook `examples/quadratic_pzt_sparse_study.py`.

## Where the 280 GiB forecast came from

The 9/64/3 mesh contains 15,687 P2 nodes and 9,792 tetrahedra per PZT.
The three projected support sizes are 41,361, 41,346, and 41,346 mechanical
DOFs. A single dense FP64 matrix on the largest support is 12.75 GiB.
Storing stiffness, consistent mass, and its inverse costs 38.24 GiB per
body, about 114.65 GiB for three bodies. The old forecast additionally
includes another dense-patch-sized workspace, plate storage, integration
vectors, and a 15% margin, giving 279.94 GiB. It was a conservative forecast,
not an observed allocation. Dense electrical condensation and dense interface
projection also inflated the host forecast to about 586 GiB.

## Equivalent operator formulation

The new implementation calls the ORIGINAL P2 element assembly: the same
quadratic displacement and potential basis, full consistent mass, quadrature,
material coefficients, electrode constraints and interface interpolation.
No precision reduction, mass lumping, incomplete factorization, damping,
model reduction, gain correction or mesh reduction is used.

Let `T` map retained plate/body coordinates to the PZT nodal displacement,
`B_i` be the coupling to interior electrical nodes, and `D_ii` their
dielectric stiffness. Short-circuit stiffness is applied as

```text
u = T x
y = T.T [K u + B_i solve(D_ii, B_i.T u)]
```

The dense electrical Schur complement is never constructed. Electrode shape
potential, capacitance, and force/readout vector are obtained from the same
grounded-lower/equipotential-upper electrical problem as the dense reference.
Open circuit adds the same `g (g.T x) / (C + C_input)` rank-one action.

PZT consistent mass remains a sparse matrix. The three Cartesian mass blocks
are identical for the supported isotropic host mass; one projected scalar
mass matrix, including the host mass on the contact support, is LU-factored
and reused for three right-hand sides. The mass inverse is applied by solves,
never explicitly formed. The current three supports were verified disjoint.
Overlapping supports are rejected, not independently solved or lumped. The
dense merged-support backend remains available for that case.

Full FP64 sparse LU setup runs on the CPU. GPU mode uploads the complete LU
factors and uses GPU triangular solves; component coefficients and interface
maps remain sparse on the GPU. An explicit `cpu` factor mode performs only
the small component solves on the CPU, transferring their RHS/solutions;
the plate and sparse matrix products remain on the GPU. This was faster on
the local machine. There is no automatic backend switch.

The body frequency bound avoids a dense generalized eigenproblem on the
entire patch: pointwise electrical maximization gives the stiffness bound
`C + e.T eps^-1 e`, and the maximum element generalized eigenvalue bounds the
assembled body against its consistent mass. Local dense eigenvalues have a
numerical safety margin, not an interval-arithmetic certificate. This is a
construction bound, not validation of the requested LTS time pair.

## Memory measurements and forecasts

| Quantity | Dense reference | Sparse implementation |
|---|---:|---:|
| Largest body's stored matrices/factors and maps | 38.24 GiB for three matrices | 165.36 MiB conservative CPU array/factor accounting |
| Three bodies, matrix/factor accounting | about 114.65 GiB | about 497 MiB (duplicates the shared component) |
| One full-size body's GPU pool after warmup | cannot fit locally | about 219 MiB |
| Complete-study GPU admission forecast | 279.94 GiB | 31.63 GiB |
| Complete-study data-based estimate | not measured | about 11.71 GiB, including declared plate storage and recurrence workspace |

The 31.63 GiB admission forecast conservatively includes per-factor budgets,
duplicate shared components, transpose copies, and vectors. It remains larger
than the data-based estimate. Factor fill, GPU library workspace, allocator
pools and a full-run peak have not been measured together. SuperLU does not
provide a hard workspace cap: pre-factor estimates and post-factor byte checks
are retained, but are not mathematical peak-memory guarantees.

The one-body GPU measurement does not establish that the entire plate fits
on the local 8 GiB GPU. The complete study remains a cloud-sized workload.
No cloud allocation was inspected or started during this task.

## Accuracy and cost checks

25 lightweight CPU tests passed, covering the original P2 tests, study
report/restart behavior, and sparse equivalence. Sparse tests cover both PZT
faces and poling signs, open/driven ports, added readout capacitance,
stiffness and mass products, charge/readout vectors, capacitance, reciprocity,
positive energy, consistent mass solves, and six-frequency component voltages.
Dense voltage differences at 50/100/150/200/250/300 kHz were below the test's
`1e-11` relative threshold. No full plate propagation was involved.

Small actual GPU checks used the unchanged `LocalActions` and original
`split_lf2_step` for 16 steps. GPU sparse versus dense action differences
were below `6e-16`, recurrence state differences about `1e-16`, and measured
unforced energy drift about `5.6e-15`. CPU-factor hybrid mode also passed.

The full-size component audit assembled one shared PZT and factored all
three physical contact mass matrices in sequence. Mass residuals were below
`9.3e-16`. Full-size GPU stiffness and inverse-mass actions matched CPU sparse
reference actions within `6.3e-16`. These full-size comparisons validate the
operator implementations, not independent dense full-size matrices or full
waveform behavior. CPU RSS at the end was about 1.1 GiB; it is not a measured
peak. Shared component assembly/bound took 20.69 s, and the three mass
factorizations took about 0.88/1.01/0.65 s on this computer.

Five warmed full-size component actions on the RTX 3060 Ti measured:

| Action | GPU factor solves | CPU factor solves, GPU products |
|---|---:|---:|
| Stiffness | 99.17 ms | 3.22 ms |
| Consistent mass inverse | 265.46 ms | 14.96 ms |

These measurements include wrapper/library overhead. They do not predict
H100/H200 timings. GPU sparse triangular solves have dependency/analysis
overhead; less memory does not guarantee greater speed. Tiny dense action
pairs were also faster than tiny sparse pairs (about 1.3 versus 11.7 ms in
one GPU-factor measurement). A full-size dense timing comparison was not
attempted because it exceeds local memory. No full-study speedup is claimed.

## LTS preservation and explicit selection

SHA-256 hashes before/after match for the original LTS files, optimized LTS
file, original consistent mass merge, dense piezoelectric implementation,
quadratic dense implementation and original GPU PZT backend. Git diff also
shows no task edits in those protected files. Evidence is in
`docs/evidence/quadratic-pzt-sparse/`.

The sparse storage adapter subclasses the existing storage interface so the
original LTS actions call it without changing the recurrence. This backend
requires `--execution-backend original` and the previously requested
`--skip-stability-checks` profile. The existing CUDA-graph adapter assumes
dense arrays, and cached stability normalization assumes dense Cholesky
blocks; they are rejected for sparse mode. No solver files were patched to
make them accept sparse objects. The existing fine 5.340576171875 ns / coarse
10.68115234375 ns pair is retained, and remains unqualified for this P2 mesh.

A fresh metadata-only plan was prepared and validated at
`outputs/quadratic-pzt-sparse-audit/final-prepared-study`. It reuses the saved
linear 6/32/3 baseline, contains one quadratic 9/64/3 broadband impulse solve
and six frequency reconstructions, and retains all reporting/restart behavior.
This compares both element order and mesh, not pure P2 mesh convergence.
Preflight fits the declared 64 GiB device budget; it does not certify current
hardware availability, full-run memory, stability, or waveform convergence.

Preparation and preflight only, with the archived baseline paths supplied:

```powershell
python -m examples.pzt_mesh_study --prepare --root NEW_STUDY_FOLDER --baseline BASELINE_FOLDER --assembly-sources ORIGINAL_ASSEMBLY_FOLDER --pzt-storage sparse --pzt-factor-backend gpu --execution-backend original --skip-stability-checks
python -m examples.pzt_mesh_study --preflight --root NEW_STUDY_FOLDER
```

Choose `--pzt-factor-backend cpu` explicitly for the measured hybrid option.
Dense selection remains `--pzt-storage dense`. The component audit is
`python -m examples.quadratic_pzt_sparse_check`; its explicit GPU and target
options run component checks only. Windows dependency paths and the baseline
plan path are documented by its CLI/environment errors; no credentials or
machine-specific paths are embedded in production operators.

No full simulation or RunPod action was performed. Earlier preparation files
and receipts remain intact; use the final prepared folder for current hashes.

## References

- [PETSc virtual Schur complements](https://petsc.org/main/manualpages/KSP/MatCreateSchurComplement.html): apply a Schur action through inner solves without forming the dense matrix.
- [CuPy SuperLU](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.SuperLU.html): CPU factor object with GPU solve support.
- [CuPy sparse LU](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.splu.html): factorization is on the CPU; GPU solve support does not imply GPU factorization.
