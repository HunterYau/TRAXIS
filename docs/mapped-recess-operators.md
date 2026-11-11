# Mapped recess elastic operators

`traxis.mapped_tensor` applies three-dimensional isotropic elastic stiffness
on the explicit conforming recess connectivity. The operator evaluates the
inverse geometric Jacobian at every displacement GLL point, forms physical
strain/stress and applies the weak transpose derivative. Production action
uses tensor contractions and gather/scatter without storing dense element
stiffness matrices. Geometry degree must not exceed displacement degree so
physical affine fields and rigid motions remain representable.

Positive GLL mass is assembled from density times the positive mapped volume
weights. Geometry and displacement nodes are checked for consistency. Source,
connectivity, Jacobian, vector and workspace allocations are screened before
operator construction. Batched CPU action supports at most 32 row vectors.
Dense assembly is restricted to a separate small audit path.

## Conservative explicit stability bound

For Lamé constants lambda and mu, C=max(2*mu,3*lambda+2*mu) bounds isotropic
strain energy by C times the squared physical displacement gradient. Applying
weighted Cauchy-Schwarz to each gradient row gives a positive diagonal
majorant H with K <= H in quadratic-form order. Absolute inverse-Jacobian
and reference-derivative contractions compute this bound without forming a
dense stiffness matrix. Shared-node contributions scatter into global H.

With positive diagonal mass M, the largest generalized eigenvalue is bounded
by max(H_i/M_i). The returned stable step is 2/sqrt(bound). This is a
conservative bound for the represented discrete GLL system, not an accuracy
criterion or a chosen recording step. It is recomputed for each geometry;
the Cartesian intact-plate bound is not reused.

## Native reference

`examples.recess_operator_reference` compares every native element column and
GLL mass against the tensor contractions. Native geometry uses the checked
VectorH1 deformation. Each local H1 basis is transformed into the shared
physical nodal basis, including checks on component ordering, shared traces
and conforming coefficients.

Global native bilinear-form applications use `nonassemble=True` to avoid
large global matrices. A sparse nodal evaluation transform maps the custom
operator's residual to the native dual basis. Random, affine and six rigid
fields check global stiffness/mass, work and shared assembly. Native local
spectra check six rigid modes and the diagonal energy majorant. Small fixtures
also retain complete connected matrices and independently measured extremal
eigenvalues. Dense matrices are verification artifacts only.

The installed native application behavior was checked directly, consistent
with NGSolve's [symbolic-integrator documentation](https://docu.ngsolve.org/latest/how_to/symbolic_integrators.html)
and [matrix-free operator tutorial](https://docu.ngsolve.org/nightly/i-tutorials/unit-3.5.1-dgapply/dgapply-scalar.html).

```powershell
.\.venv\Scripts\python.exe -m examples.recess_operator_reference --order 3 --angular-cells 4 --output outputs/mapped-recess
.\.venv\Scripts\python.exe -m examples.recess_operator_reference --order 3 --angular-cells 4 --intact --output outputs/mapped-intact
```

Each run freezes its core/driver/runtime identities and source archive before
the worker starts. Element proofs are written separately to bound memory.
The parent validates terminal receipts and archive hashes. Operator checks
do not propagate a waveform or accept a research data bank.

## Preflight checkpoint

The first 17-test group and final 25-test group pass without skips. The final
group includes mapped action, affine strain energy, rigid fields, an
independent spectral check of the diagonal bound for positive and negative
lambda, allocation rejection, geometry and existing tensor/storage checks.
The groups overlap and do not increase the last complete integrated total.

The initial connected Q3 recess reference passes full matrix, native global
application and spectrum checks. Adding the full-matrix archive digest to
the report tightens provenance; the source-bound repeat also passes. This
is not a numerical failure or a change in the acceptance tolerance.
The original passed result remains retained separately.

See `evidence/mapped-recess-preflight-debug.json` for exact preflight receipts.
Q3/Q4/Q5 selected-geometry paired controls and their independent raw-array
audit are recorded below. Mapped GPU capacity, physical
disk integration, complete paired waves and geometry/space/time convergence
remain necessary. PZT/electronics and physical specimen agreement remain
open. No research training runs during these checks.

## Completed native cases and raw-array audit

All eight native cases pass with one fixed core/driver/runtime identity:
Q3/Q4/Q5 on the selected four-elements-per-quarter-circle geometry, each with
a recess and paired intact control, plus two coarse Q3 connected fixtures.
The plate is 40-by-30-by-3 mm with an off-center 16 mm diameter, 1.5 mm deep
backside recess. Material defaults are 68.9 GPa, nu=0.33 and 2,700 kg/m³,
bound by the retained configuration source. Geometry stays cubic throughout.

| Degree | Recess coefficients | Intact coefficients | Conservative step (ns) | Recess / intact worker (s) |
| --- | ---: | ---: | ---: | ---: |
| Q3 | 7,212 | 9,597 | 12.993378 | 7.182 / 9.936 |
| Q4 | 15,855 | 21,627 | 7.408059 | 26.546 / 31.290 |
| Q5 | 29,538 | 40,953 | 4.763678 | 58.460 / 92.653 |

The bounds coincide for these paired fixtures. They are conservative
discrete stability limits, not selected recording steps. The earlier intact
plate's 7.62939 ns step is above the Q4 mapped bound here; it cannot be
carried over without checking. Future experiments must freeze an appropriate
common step after accounting for actual geometry and thin regions.

Every one of 511 native element matrices passes its mass, six-rigid-mode and
diagonal-majorant spectrum checks. The two coarse connected systems also
pass full-matrix comparisons and smallest/largest generalized-eigenvalue
checks. For the coarse recess, the diagonal bound returns 21.591 ns versus
53.670 ns from the measured largest native eigenvalue, a factor 6.179 in
eigenvalue bound. The six rigid eigenvalues are identified using normalized
tolerances; tiny nonzero or negative floating-point values are retained.

A separate auditor reconstructs physical shape gradients from retained
native Jacobians, then assembles stiffness through a six-component strain
matrix and constitutive matrix. This differs from the production
gradient/stress/divergence action. It checks the maximum relative discrepancy
of each column, transforms the raw native hierarchical matrices independently,
reassembles positive mass and recomputes the diagonal majorant. It then
scatters native element actions, verifies the sparse native basis transform,
and compares the resulting global actions with native bilinear-form Apply.
The saved full matrices and spectra are also checked independently.

The audit covers 200,352 element columns. Maximum discrepancies are:

| Check | Relative discrepancy |
| --- | ---: |
| Independent strain assembly / native, worst column | 3.51e-13 |
| Mapped action / native, worst column | 3.53e-13 |
| Native GLL mass / independent mapped mass | 7.70e-15 |
| Reassembled global stiffness / native Apply | 4.33e-14 |
| Reassembled global mass / native Apply | 8.74e-15 |
| Physical affine strain energy | 2.40e-14 |

The normalized global rigid-field residual is below 1.14e-18. All numerical
comparison limits remain 1e-10; normalized zero/majorant limits remain 1e-12.
The producer's whole-matrix relative norms and the auditor's maximum-column
norms are different metrics. No acceptance limit was adjusted to obtain a pass.

Peak combined RSS across native jobs is 400.31 MiB, system RAM 56.3% and
minimum available RAM 7,092.78 MiB. GPU observations are 857 MiB / 51 C,
including desktop activity; these computations are CPU-only. CPU temperature
is unavailable. The independent audit completes in 55.625 sampled elapsed
seconds at 214.16 MiB peak RSS, with 50.2% system RAM and at least 8,081.67 MiB
available. No resource guard is relaxed and no training updates run.

Raw cases are the `outputs/mapped-recess-*` and `outputs/mapped-intact-*`
directories dated 20261110. The coarse recess's `-v2` directory carries the
final full-matrix digest binding; its original passed receipt is preserved.
Independent CSV and PNG/PDF artifacts are in
`outputs/mapped-recess-native-audit-20261110`. The final figure was visually
checked. Compact evidence is `evidence/mapped-recess-operators-debug.json`,
fingerprint `ff800efac3ed91ead352cf8ded8ebf5d0177ba38e9767604ab48c357f486cbb3`.

These are small operator and stability checks on a fixed represented
geometry. They do not establish full-size recess cost, waveform convergence,
adequate source/receiver integration, PZT voltages or physical accuracy.
