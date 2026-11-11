# Circular-wall shape resolution

The initial conforming geometry study found a cubic quarter-circle with
0.0188% area error but 0.309% radius error. The next frozen study therefore
checks both area and radius before selecting geometry for later wave work.
It does not relax a wave criterion or accept a propagation discretization.

## Declared screen

Keep geometry cubic, so its degree is no higher than the planned Q3 coarse
displacement space. Test 1, 2, 4 and 8 elements per quarter-circle. Require
area relative error below 1e-5 (0.001%) and radius relative error below
1e-4 (0.01%). Choose the first passing subdivision; retain coarse failures.
The limits and sequence are written into the plan before evaluation.

| Elements per quarter-circle | Area relative error | Radius relative error | Result |
| --- | ---: | ---: | --- |
| 1 | 1.87570e-4 | 3.09352e-3 | Fails both |
| 2 | 3.06031e-6 | 1.96961e-4 | Fails radius |
| 4 | 4.83365e-8 | 1.23672e-5 | Passes |
| 8 | 7.57299e-10 | 7.73840e-7 | Passes |

Four elements per quarter-circle are selected. Maximum represented radius
error is approximately 0.124-0.248 micrometers for the planned 10-20 mm radii.
This small geometric error is not a bound on scattered-wave error.

## Physical domains and independent native checks

All nine planned diameter/depth combinations and their paired intact
controls pass geometry volume, boundary and determinant checks in the
500-by-500-by-3 mm box. The represented-volume discrepancy is below 1.34e-15.
The minimum polynomial determinant bound is 4.85e-10 and maximum sampled
Jacobian condition number 462.21. These coarse block domains are not a
full-size wave-mesh capacity benchmark.

Native comparisons use the selected subdivision on an off-center 16 mm
diameter, 1.5 mm deep recess in a 40-by-30-by-3 mm plate:

| Fixture | Hexahedra | Q3 vector coefficients | Native Jacobian discrepancy |
| --- | ---: | ---: | ---: |
| Paired intact | 96 | 9,597 | 1.53e-14 |
| Recess | 64 | 7,212 | 1.53e-14 |

Native coordinates differ by at most 4.17e-17 m and native volume by
2.95e-14 relative. A separate auditor recomputes the shape selection with
Legendre interpolation and quadrature. It also fits the raw native coordinate
arrays and differentiates them independently; derivative discrepancies are
below 1.23e-13 and resulting volumes agree within 1.23e-15. No native elastic
operator or time trajectory is evaluated by these geometry checks.

The worker completes in 34.110 monitored seconds with 397.24 MiB peak combined
RSS, 54.2% peak system RAM and 7,433.14 MiB minimum available RAM. GPU
observations are 873 MiB / 51 C, including desktop activity; this job is
CPU-only. CPU temperature is unavailable. No training or data generation runs.

```powershell
.\.venv\Scripts\python.exe -m examples.recess_shape_reference --output outputs/recess-shape
```

The plan binds all core sources, four wave drivers, both geometry-analysis
drivers and the native runtime, with retained source snapshots and terminal
resource receipts. Raw data are `outputs/recess-shape-native-20261110`;
independent CSV and PNG/PDF artifacts are `outputs/recess-shape-audit-20261110`.
The final figure was visually checked. Compact evidence is
`evidence/recess-shape-refinement-debug.json`, fingerprint
`f19fd8523765df284342c2eecc0138320a074c8b2487a457865ea249a657633c`.

The subsequent mapped elastic operators, positive mass and conservative
stability bound pass small native checks in mapped-recess-operators.md.
Nonrectangular physical footprint integration and GPU propagation remain.
Geometry selection remains separate from wave space/time refinement, representative
defect/frequency coverage and actual bonded PZT/physical agreement.
