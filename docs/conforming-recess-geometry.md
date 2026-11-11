# Conforming circular-recess geometry

`traxis.recess_hex` builds explicit connected hexahedra for one circular
flat-bottom backside recess and its paired intact control. It is geometry
infrastructure, not a mapped elastic operator or wave backend.

## Material topology

A central square, four square-to-circle blocks and four circle-to-plate-edge
blocks cover the planar plate. Matched layers meet at the recess-floor plane.
The defective mesh omits the five inner blocks below that plane; the intact
control fills them. At the coarsest subdivision this gives 13 defective or
18 intact hexahedra. The upper surface remains flat.

Node identity comes from vertices, oriented edges, faces and vertical lattice
indices, not rounded physical coordinates. Shared faces must have identical
displacement and geometry node sets. Every unmatched face must be a declared
top, bottom, outer edge, recess wall or recess floor. The material face graph
must be connected. Subdivisions match across block and depth interfaces.

The geometry polynomial degree is independent of the displacement degree.
Changing Q3/Q4/Q5 on fixed geometry preserves the geometry controls and maps
exactly. Changing the angular subdivision also changes the represented circle
and must be recorded as geometry refinement. Allocation is screened before
creating grids, dictionaries or connectivity. The reference implementation
supports one strictly interior recess; it does not select a wave resolution.

## Independent checks

The 90-case study covers all 20/30/40 mm diameters and 0.75/1.50/2.00 mm
depths in the actual 500-by-500-by-3 mm bounding box, both defective and intact,
through geometry degrees one to five. These are coarse geometric domains,
not full-size propagation-capacity measurements.

Volume quadrature, closed-boundary normal balance, the divergence-theorem
volume and top/floor areas agree with the represented polynomial geometry.
The circular reference separately integrates its closed polynomial boundary
and computes radial extrema. All geometric Jacobians have positive sampled
determinants. A Bernstein-coefficient lower bound also checks the structured
map's determinant polynomial between nodes. This is floating-point polynomial
verification, not interval arithmetic with certified rounding bounds.
The minimum bound is 1.94e-9 and maximum sampled condition number 637.26.

Ten small native fixtures cover intact/recess geometry at every degree.
Native H1 nodal interpolation creates a VectorH1 geometry deformation on
the straight base mesh. Native physical coordinates, full Jacobians and
integrated volume are compared against the independent block map.
Maximum coordinate, Jacobian and volume discrepancies are 2.23e-16 m,
2.15e-14 relative and 3.02e-14 relative. A separate auditor fits Legendre
polynomials to retained native coordinates and differentiates them, finding
Jacobian discrepancy below 7.30e-14 and volume discrepancy below 5.22e-15.
This checks native geometry, not elastic stiffness or a transient.

At one element per quarter-circle, shape errors are:

| Geometry degree | Circle area relative error | Radius relative error |
| --- | ---: | ---: |
| 1 | 3.63380e-1 | 2.92894e-1 |
| 2 | 1.17848e-2 | 1.07814e-2 |
| 3 | 1.87570e-4 | 3.09352e-3 |
| 4 | 1.73825e-6 | 9.08731e-5 |
| 5 | 1.04669e-8 | 1.52409e-5 |

The cubic area error is much smaller than its radius error. Area agreement
alone does not establish circular-wall accuracy. The subsequent
[shape-resolution study](recess-shape-refinement.md) checks both quantities.

## Tests, failures and resources

The six new geometry tests plus the existing tensor and storage checks pass
all 19 tests without skips in 1.876 unittest seconds. They cover physical
volume/boundaries, refined shared interfaces, fixed geometry across solution
degrees, shape refinement, finite-difference Jacobians and allocation rejection.

The initial expected defective Q3 count was calculated incorrectly as 1,596.
The annulus has 8 vertices, 12 edges and 4 faces, giving 4p²+4p planar nodes.
The correct defective total is 1,488, versus 1,848 for the paired intact mesh.
The independent count was corrected; no mesh change was needed. The failed
test log remains retained.

The first native study stopped while constructing a report with a duplicate
`passed` keyword. Correcting that merge and repeating unchanged numerical
criteria gives the completed 90+10-case result. The initial source archive
and failure receipt remain retained. The first auditor assumed 100 archive
members; the corrected audit checks the exact expected inventory of 90.
A subsequent passing audit exposed a complex-root dtype cast warning. The
final auditor checks negligible imaginary parts and uses real roots explicitly;
numerical metrics are unchanged. All attempts remain separately recorded.

The native study completes in 49.067 monitored seconds. Peak combined RSS is
345.49 MiB, system RAM 57.3%, minimum available RAM 6,929.64 MiB and GPU
observations 873 MiB / 50 C. The computation is CPU-only; GPU observations
include desktop activity. CPU temperature is unavailable. No training runs.

```powershell
.\.venv\Scripts\python.exe -m examples.guarded_regression --output outputs/recess-tests --max-seconds 120 --max-rss-mib 768 tests.test_recess_hex tests.test_tensor_hex tests.test_storage_recovery tests.test_live_progress
.\.venv\Scripts\python.exe -m examples.recess_geometry_reference --output outputs/recess-geometry
```

Raw native receipts and sources are in `outputs/recess-geometry-native-20261110-v2`.
The final independent CSV/PNG/PDF audit is `outputs/recess-geometry-audit-20261110-v3`.
Its comparison figure was visually checked. Compact evidence is
`evidence/conforming-recess-geometry-debug.json`, fingerprint
`e44722949c80b1e4b3266b8ca3bea12d75f7b2594a96fe80d0ebf6011e8263d2`.

Mapped stiffness/mass, stability bounds, physical disk integration, GPU
capacity and complete paired wave/space/time studies still remain. See
[the implementation design](tensor-recess-design.md). No geometry result
accepts a research waveform bank or substitutes for PZT/physical calibration.
