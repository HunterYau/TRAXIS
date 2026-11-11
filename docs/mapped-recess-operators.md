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
audit are recorded separately as they complete. Mapped GPU capacity, physical
disk integration, complete paired waves and geometry/space/time convergence
remain necessary. PZT/electronics and physical specimen agreement remain
open. No research training runs during these checks.
