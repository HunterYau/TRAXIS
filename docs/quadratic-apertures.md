# Quadratic circular mechanical apertures

The archived native P2 waveform measurements below predate the
[mass quadrature correction](elmer-mass-quadrature.md). The analytic surface
integrals remain unchanged; corrected native results require regeneration.

P2 mechanical sources and receivers now integrate their actual circular
footprint over straight-sided quadratic tetrahedral top faces. Signed nodal
weights preserve area, centroid and quadratic moments under remeshing. A
native Elmer pilot verifies the source and receiver interfaces, including a
truncated Hann burst and reciprocal traces. These checks do not establish
finite-plate spatial convergence or bonded-PZT voltage accuracy.

## Geometry and integration

The original [P1 aperture method](mechanical-patches.md) needs area and first
moments. P2 also needs second moments. For a homogeneous polynomial f of
degree p, Green's theorem gives

```
integral_D f dA = 1/(p+2) * integral_boundary f * (x dy - y dx).
```

After centering/scaling the disk to unit radius, the implementation integrates
`1, x, y, x², xy, y²`. Each triangle edge is split at circle intersections.
Interior segments and oriented circular arcs have analytic contributions.
For a segment p→q with c=cross(p,q), the x² contribution is
`c*(px²+px*qx+qx²)/12`; xy contributes
`c*(2px*py+px*qy+qx*py+2qx*qy)/24`. The y² formula follows by symmetry.
On an arc with signed angle dtheta, the contributions are
`(dtheta+qx*qy-px*py)/8`, `(qy²-py²)/8`, and
`(dtheta-qx*qy+px*py)/8`. Orientation and radial-edge cancellation follow
the P1 construction. This is an extension of the same boundary-integral
derivation, checked independently rather than accepted from the formulas alone.

Transform the moment matrix into barycentric product integrals Gij, and let
li be the linear shape integrals. The six quadratic integrals are
`2*Gii-li` for vertices and `4*Gij` for edges 01/12/20. Their sum gives the
actual intersection area. Multiplication by radius² restores m²; normalization
by area gives unit-force/average-displacement weights. Negative vertex
coefficients are legitimate and are retained. They are equivalent discrete
loads, not a claim that the physical traction changes sign within a uniform
patch.

`QuadraticTetMesh` bounds node/element counts, checks finite geometry, rejects
unused nodes and duplicate/degenerate tetrahedra, and requires straight
midpoints shared conformingly by neighboring elements. Its arrays are copied
and read-only. The native Elmer loader also checks declared counts, sequential
IDs, body/type 510 and the solver's edge-node ordering. Curved quadratic
geometry is unsupported by this helper and fails explicitly.

The top-face integrator identifies edge nodes from their physical midpoint
coordinates. The resulting work-conjugate vector is used both for applied
force and area-averaged displacement. Its report records area, centroid,
second moments about the requested center, direction, support/negative-weight
counts and edge clipping. A clipped disk uses its actual intersection area;
it is not silently treated as a complete sensor aperture.

Tests cover analytic full/half/quarter disks, all vertex permutations, whole
triangles, zero overlap, SI translation/scaling, quadratic-field averages and
mesh/edge-node permutations. General nonsymmetric intersections are checked
with independent adaptive x-slice / Gaussian y-quadrature, split at every
boundary switch. Malformed, curved and nonconforming meshes are rejected.

## Native load/probe check

```powershell
.\.venv\Scripts\python.exe -m traxis verify-quadratic-patches --elmer-home outputs/reference-tools/elmer-26.1/ElmerFEM-nogui-nompi-Windows-AMD64 --output outputs/quadratic-aperture-reference
```

Use the existing solver from [elmer-reference.md](elmer-reference.md) and a
fresh output directory. Defaults are 60 seconds and 512 MiB combined sampled
monitor/external-tree RSS, plus system memory and best-effort temperature/GPU
guards. The command performs three short CPU transients, with no model
training, dataset generation or global installation changes. Completed failed
screens return 2; operational failures return 1 and retain the partial report.

The fixed pilot is a free 40 x 30 x 3 mm block with 75 P2 nodes and 24 tets.
Two off-grid, asymmetric 3.2 mm radius apertures use a nominal 1 N force
scale multiplying the unnormalized Hann-windowed sine. The plan's
`peak_patch_force_n` names that scale; the actual pulse maximum is lower.
A 100 kHz two-cycle Hann pulse is evaluated natively through MATC and
truncated at 20 microseconds. The two transmit/receive roles are swapped on
the identical mesh/time grid. Seventeen union-support nodes are recorded;
checked named z-displacement columns are combined with the receiving weights.
Traces retain meters, with 128 samples at 0.5 microseconds and the known zero
initial condition. No amplitude fitting, peak normalization, interpolation or
time shift is applied.

A third run applies a uniform volume version of the same burst. Its rigid
translation is compared with independent scalar Newmark kinematics driven
by the existing Python Hann function at every time sample. This checks the
native envelope, cutoff and units. The analytic time-order validation remains
the separate [transient study](elmer-transient.md); a common discrete source
check does not replace it.

## Measured results, 2026-11-04

| Quantity | Measured result |
| --- | --- |
| Full-disk relative area error | 2.22e-16 for either patch |
| Maximum centroid error | 1.54e-18 m |
| Maximum centered second-moment error | 6.36e-21 m² |
| Negative coefficients | 3 / 4 in the two apertures |
| Relative reciprocal-trace L2 difference | 5.39e-13 |
| Relative native/Python source-interface difference | 1.27e-12 |
| Reciprocity reference RMS | 1.03e-9 m |
| Elapsed time | 3.78 s |
| Sampled combined RSS / system RAM | 116.2 MiB / 62.4% |
| Observed GPU memory / temperature | 921/8,192 MiB / 47 C |

The predeclared reciprocal/source interface tolerance was 1e-8. Area,
centroid and second moments were screened before solving at 1e-10 relative,
1e-13 m and 1e-16 m² respectively. These tolerances were not fitted to the
observed traces. GPU values describe the system; all computation used CPU.
CPU temperature was unavailable.

The full 168-test suite passed. A coverage correction made the new permutation
test start from the original vertices for each permutation; the corrected
12-test aperture/P1/native subset also passed, with no runtime-code changes.
The full suite took 95.02 seconds with sampled combined RSS 783.2 MiB and
system RAM 65.8%. GPU usage stayed at 921/8,192 MiB and 47 C; CPU temperature
was unavailable. The sampled maximum applied force was 0.8602 N for the
nominal 1 N unnormalized burst scale.
The compact record is
[evidence/quadratic-aperture-debug.json](evidence/quadratic-aperture-debug.json).
Full native inputs, scalar outputs, weights and hashes remain under
outputs/quadratic-aperture-reference.

This pilot uses a deliberately coarse fixed mesh. Reciprocity and exact load
moments can hold even when bending dispersion is wrong. Existing local bank
generation still uses P1 elements; this addition does not upgrade or promote
those banks. Finite-plate mesh/time refinement, recess scattering, supports
and actual PZT/adhesive/electrical validation remain required before research
data generation.
