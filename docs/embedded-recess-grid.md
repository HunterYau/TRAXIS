# Recess grid with a Cartesian exterior

The full 500 by 500 by 3.175 mm plate now has a local curved recess grid
joined to a Cartesian exterior. This reduces the cell-size imbalance of
the earlier globally curved grid. It is a geometry/operator/capacity
checkpoint; these jobs completed no full-size wave steps.

## Construction

`traxis.embedded_recess.make_embedded_recess_mesh` embeds nine local planar
blocks in eight rectangular exterior blocks. The default patch surrounds
the recess by a rectangular margin and contracts near plate edges. Integer
vertex/edge/face identities join the blocks without rounding coordinates.
The extrusion retains the shared floor and top layers while omitting material
below the recess. Geometry degree is independent of displacement degree.

Per-block subdivision counts allow unequal rectangular widths/heights.
Shared geometry and displacement nodes, closed boundaries and connected
material are checked. Closed-form node counts and conservative byte screens
reject oversized requests before allocation. The native geometry reference
now starts from the actual cell corners and accepts the new block topology.

The full-size case uses a recess centered at (255,245) mm with radius 15 mm
and depth 1.5 mm, cubic geometry, 12 angular cells, radial counts (2,6),
one layer below/above the floor partition and 5 mm exterior spacing.
Its local patch spans 60 by 60 mm. Corner edges range from 1.25 to 5 mm,
compared with 0.234375 to 10.800764 mm in the earlier global curved grid.
These sizes are capacity choices; wave accuracy still requires refinement.

## Independent numerical checks

Six bounded native/GPU jobs cover paired intact/recess Q3, Q4 and Q5 cases.
NGSolve provides independent geometry, hierarchical element assembly and
global operator application. These fresh references use NGSolve; the earlier
Elmer checks remain separate evidence.

All 348 local elements and 140,940 local stiffness columns pass. Maximum
element stiffness and mass discrepancies are 1.46e-14 and 1.34e-14 relative.
The GPU/global CPU field discrepancy is at most 2.51e-14. Independent native
disk integration agrees within 2.14e-15 for all 24 load/probe covectors,
including a disk crossing the local-to-Cartesian patch corner. Signed
weights retain unit total force and physical first moments.

An independent saved-array audit checks the hierarchical-to-nodal transform,
element stiffness/mass, native/GPU actions, disk vectors and artifact/source
hashes. It does not infer correctness from the producer's pass flags alone.
Full-size whole-cell geometry bounds and their retained proof are recorded
in the compact evidence receipt linked below.

The geometry audit checks all 41,296 represented cells. It verifies every
cell's tensor-extrusion structure, bounds the 10,384 unique planar determinant
polynomials and scales each by its positive half-height. Every bound is
positive: minima are 3.27148e-10 m^3 for recess and 2.92969e-10 m^3 for intact.
This is a floating-point Bernstein bound, not an interval-arithmetic proof.
The audit finishes in 75.958 seconds at 160.50 MiB peak RSS; its PNG wireframe
and capacity plot was visually inspected. A matching PDF is retained.

## Full-size capacity measurements

| Case | Hexahedra | Displacement coefficients | Conservative stable step (ns) | Median GPU stiffness (ms) | Forecast steps for 500 us |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recess | 20,528 | 4,462,575 | 5.509508 | 4.03456 | 113,441 |
| Intact | 20,768 | 4,507,515 | 5.438163 | 4.08461 | 114,929 |

The step forecast uses 80% of the conservative stability limit. It is not
an accuracy-qualified time step. Compared with the old grid's 427,511 and
427,908 steps, these forecasts are about 3.77 and 3.72 times smaller.
Measured stiffness timings project 7.63 and 7.82 minutes of operator work
per 500 us window, excluding integration, sampling, guards and cooling.
No full propagation has been timed on these meshes.

Full CPU/GPU smooth-field actions agree within 1.17e-15. The retained-array
audit compares mass directly with analytic stock-minus-circular-recess mass
within 1.09e-13, and normalized centroid error stays below 2.03e-13.
Each actual corner disk takes about 14.3-14.7 seconds to integrate; the
workers finish in 80.15 and 80.85 seconds. Operator-only projection is not
complete waveform-bank throughput.

Reference and capacity jobs peak at 1,141.84 MiB process-tree RSS, 65.2%
system RAM and 1,469 MiB observed GPU memory / 52 C. At least 5,643.38 MiB
system RAM remains available in those samples. GPU readings include desktop
activity; CPU temperature is unavailable.

## Tests, retained failures and current scope

The final focused suite passes 42 tests without skips in 63.920 seconds.
It covers the new mesh, old compact/reference meshes, mapped CPU/GPU actions,
circular ports, seeded PZT randomization, native pulses and external tasks.
Peak test-process-tree RSS is 505.97 MiB, system RAM 61.7%, GPU observations
923 MiB / 50 C and minimum available RAM 6,209.84 MiB. CPU temperature is
unavailable. The earlier 17-test mesh check is overlapping evidence, not an
additional independent suite total.

A separate compact-intact comparison attempt hit the unchanged 360-second
wall-time limit after logging mesh construction. It has no accepted report
and was not promoted or silently retried. The earlier successful compact
capacity evidence remains historical. The first saved-array audit stopped
on a summary-field typo (`elapsed_seconds` instead of `wall_seconds`); the
corrected audit used a fresh directory and passed without changing numerical
criteria. Both failures are retained under outputs.

The initial geometry audit repeated identical planar polynomials across
layers/paired meshes and reached its 240-second wall-time limit near the end
of the intact plate. That attempt has no accepted geometry result. The
rerun checks the extrusion controls of every cell and reuses only identical
planar geometry. Its resource limits remain 240 seconds / 512 MiB, system
RAM 85%, at least 2 GiB available and GPU temperature 80 C. The failed helper
and explanation are retained alongside the successful fresh output directory.

The November 12 user instruction removes clamp calculations and retains
slight PZT position, effective footprint and coupling variation. Existing
drift, drive-force and noise variation also remain. External task metadata
now asks for measured mounting/coupling response rather than clamp-specific
inputs. The recipe ranges remain provisional. This metadata edit follows
the frozen numerical runs and does not relabel their source identities.

Next numerical work is complete embedded-mesh wave windows followed by
separate spatial/time comparisons and representative thickness/frequency
checks. Physical waveform comparison and PZT/electrical response remain
open. No research bank is accepted and no model training ran here.

See [the compact evidence receipt](evidence/embedded-recess-capacity-debug.json),
[PZT variation](pzt-domain-randomization.md), and
[the earlier compact-grid measurements](full-size-recess-capacity.md).

Reproduce individual checks in fresh project output directories:

```powershell
.\.venv\Scripts\python.exe -m examples.embedded_reference --order 4 --output outputs/embedded-native-check
.\.venv\Scripts\python.exe -m examples.recess_capacity --mesh embedded --order 4 --output outputs/embedded-capacity-check
.\.venv\Scripts\python.exe -m examples.recess_capacity --mesh embedded --order 4 --intact --output outputs/embedded-intact-capacity-check
```
