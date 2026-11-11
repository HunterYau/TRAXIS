# Full-size compact recess capacity

Paired 500 by 500 by 3.175 mm Q4 intact/recess systems now build and run
operator checks on this computer under the existing resource guards. The
recess is centered at (255,245) mm, radius 15 mm and depth 1.5 mm. This is a
capacity and implementation checkpoint. Full-size propagation and wave
convergence remain open.

## Compact construction

`traxis.compact_recess.make_compact_recess_mesh` constructs the same nine
planar blocks as the reference builder and extrudes them with array-based
global z-plane numbering. It retains a planar topological lookup, replacing
the per-node 3D Python dictionary. Coordinates are never merged by rounding.
Below the recess floor, only annular material is retained; the floor and
upper planes share the complete planar node set. Integer connectivity uses
int32. Geometry controls remain independent of displacement degree.

Sorted corner-face keys find neighbors. Every shared face checks its complete
displacement and geometry node sets, and the material adjacency graph must
be connected. Count and memory forecasts reject before construction. A
closed-form topology count predicts both solution and geometry nodes.

Small fixtures match the original builder under an explicit integer node
permutation for displacement degrees one through five, including unequal
radial/layer partitions. Coordinates, element connectivity, physical
boundaries, mapped actions and mass agree. Native geometry now binds target
controls by local topology, allowing either valid global numbering. Small
compact Q3 intact/recess fixtures pass native all-column, mass and spectrum
checks. Numerical tolerances are unchanged.

`mapped_tensor.build_system(..., max_batch_vectors=1)` declares a one-vector
global CPU reference budget for the large cases. Its allocation forecast
scales with that limit; larger batches reject before output allocation.
The default remains 32 for existing small references.

## Measured full-size cases

Both cases use cubic geometry, 64 angular cells, radial counts (2,32), and
one cell layer below/above the recess-floor partition. There are 12,800
planar cells. The shape partition is fixed between the paired cases.

| Case | Hexahedra | Displacement coefficients | Conservative stable step, ns | Median GPU stiffness, ms |
| --- | ---: | ---: | ---: | ---: |
| Recess | 20,992 | 4,664,847 | 1.461953 | 4.11648 |
| Intact | 25,600 | 5,543,451 | 1.460596 | 5.01760 |

Full CPU/GPU smooth-field actions agree within 1.20e-15 relative. Scalar
mass sums agree with analytic stock-minus-recess mass within 4.45e-15;
normalized mass-centroid errors are below 3.56e-13. Both physical 3 mm-radius
corner disk ports preserve analytic force and first moments. These are
mechanical covectors, not identified clamped-PZT voltage/traction models.

An additional geometry audit checks every represented cell, beyond the
integration points. It reuses the displacement-independent cubic controls
and computes floating-point Bernstein bounds for the determinant polynomial.
All 46,592 bounds are positive: minimum 1.15013e-11 m^3 for the recess and
1.02997e-11 m^3 for intact. This is a floating-point polynomial certificate,
not an interval-arithmetic proof. The audit retains every lower bound and
takes 138.630 seconds.

Array budget forecasts are 779.65/933.04 MiB on device and 630.30/731.82 MiB
on host for recess/intact. Device forecasts include sixteen global vectors
for future resident stepping. Whole process-tree measurements, including
runtime libraries and temporary proof arrays, peak at 1,170.10/1,254.00 MiB
RSS. Aggregate system RAM peaks at 59.4%, minimum available RAM 6,596.85 MiB,
and observed GPU use at 1,534 MiB / 52 C. GPU readings include desktop use.
CPU temperature is unavailable. Long-propagation thermal behavior is unmeasured.

## Cost and mesh limits

The guarded workers finish in 97.001/107.267 seconds. Geometry/mass/stability
setup dominates the first stage. Each physical disk currently scans all top
faces and takes roughly 18.5–18.6 seconds to integrate. A reusable surface
index would reduce repeated placement/remount setup work.

The smallest corner-edge chord is 0.234375 mm and the largest 10.800764 mm,
a roughly 46-fold size range. Fine central cells constrain explicit time
steps while much larger exterior cells still need wave-resolution checks.
This global nine-block partition couples central and exterior angular
subdivision and is not yet a selected research wave mesh.

At 80% of the conservative stability bound, a 500 us interval would need
427,511/427,908 steps. Multiplying by the measured median stiffness cost
gives 29.33/35.78 minutes of operator work alone. These are extrapolations;
no time step was propagated. Integration, sampling, guard/cooling overhead
and an accuracy-qualified time grid are excluded. The next mesh work should
balance central/exterior resolution and measure convergence and cost together.

## Verification, retained failures and use

The final focused integration passes 34 tests without skips in 26.911 test
seconds. It includes compact/reference topology and coordinate equivalence,
preallocation screens, declared batch rejection, small native geometry and
all-column/spectrum checks, GPU operators, mapped ports and mounting tests.
Test RSS peaks at 502.16 MiB and observed GPU use at 1,190 MiB / 52 C.
Overlapping test groups are not added to historical totals.

The initial 17-test group has eleven errors from a test-helper typo calling
the integer property `np.iinfo(...).max`. Correcting it produces a passing
17-test rerun and the final 34-test group. Native geometry/column/spectrum
checks already passed in the first group. The first capacity attempt stops
before mesh allocation because this Windows psutil version rejects a
WindowsPath. Passing `str(output)` fixes it; the original failure and the
fresh successful v2 reference are both retained. No criteria were loosened.

```powershell
.\.venv\Scripts\python.exe -m examples.recess_capacity --order 4 --output outputs/new-recess-capacity
.\.venv\Scripts\python.exe -m examples.recess_capacity --order 4 --intact --output outputs/new-intact-capacity
```

Commands require fresh project output directories and retain frozen source,
runtime/GPU identities, plans, resource histories and proof arrays. The
independent array audit recomputes state fields, CPU/GPU errors, analytic
mass/centroids, lower-material exclusion, port force/moments and runtime
extrapolations. A visually checked PNG/PDF shows measured sizes, device
timings, uneven edge sizes and operator-only forecasts.

Raw jobs are `outputs/compact-capacity-recess-q4-20261111-v2` and
`outputs/compact-capacity-intact-q4-20261111`. Audits/tools are
`outputs/compact-capacity-audit-20261111`,
`outputs/compact-capacity-geometry-audit-20261111` and
`outputs/compact-capacity-tools-20261111`. Compact evidence is
`evidence/full-size-compact-capacity-debug.json`, fingerprint
e59d94a9ffd309f4406b02d672bdec78d6f52857e8da7792bcb0328f8f44b167.

Full-size space/time/frequency convergence, measured clamp/contact/PZT and
electronics response, physical validation and accepted waveform banks remain
required before meaningful matched reconstruction/policy experiments. No
training or accepted research-bank generation ran; readiness stays false.
