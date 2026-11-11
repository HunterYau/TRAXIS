# Circular surface ports on the mapped recess mesh

`traxis.mapped_patches` connects physical disk footprints to the explicit
curved-mesh elasticity/GPU path. Loads and area-average displacement probes
use the same signed covector, preserving mechanical work conjugacy. Positions
and contact radii can come from each state's actual randomized mounting.
This remains a mechanical proxy for the removable PZT clamps.

## Integration and state binding

The block mesh has a flat top face whose physical coordinates are linear in
the first reference coordinate and polynomial in the second. At each value
of the second coordinate, the physical disk cuts a straight line in a known
interval. The implementation integrates the first-coordinate basis and
area Jacobian analytically, then uses adaptive vector quadrature in the
second coordinate. Disk/edge intersections, line tangencies and polynomial
extrema split the quadrature. Bernstein control enclosures screen candidate
faces and positive surface Jacobians in floating-point arithmetic; they are
not an interval-arithmetic proof.

Weights are divided by the analytic disk area, retaining negative basis
weights. The code checks total force and the physical first moment rather
than fitting a computed area. Disks must remain inside the plate. Surface
orientation, geometry degree and ruled/flat geometry are checked, with face
count and workspace limits before quadrature. Geometry degree must not exceed
displacement degree. Physical linear fields are then represented exactly;
general physical quadratic fields need not be represented by Q3-Q5 on cubic
curved geometry and are not used as exact moment criteria there.

`scenario_patches(mesh, scenario, state)` resolves actual positions/radii
through `Scenario.sensor_geometry`. It rejects a mesh with the wrong plate,
recess or intact/current state. Scalar coupling gains stay separate from the
unit-force covectors and must apply once on the drive/readout paths. Model
inputs continue to use nominal coordinates in the existing bank workflow.
The new mapped interface is not yet a validated bank-generation command.

## Native surface reference

Six independent native runs pass: Q3/Q4/Q5 paired intact/recess meshes with
cubic geometry and four angular cells per quadrant. Each checks four disks:
a corner source, that source shifted by (0.4,-0.3) mm with radius increased
from 3 to 3.3 mm, a disk crossing the curved interface, and a disk crossing
a curved angular vertex. There are 24 retained load-vector comparisons.

The native reference samples deformed NGSolve geometry independently, evaluates
native hierarchical shapes/Jacobians and uses Gaussian inner quadrature plus
cosine-transformed Gaussian outer quadrature. Outer orders 16/32/64/128
refine until successive vectors differ by less than 1e-11. This method does
not use producer basis antiderivatives or per-face integrals. Full native
covectors agree with mapped covectors transformed to the native basis within
3.20e-14 relative, against a declared 1e-9 limit. Maximum analytic area and
radius-scaled first-moment errors are 1.12e-15 and 1.05e-14.

## Complete small pulse windows

A separate Q4 control uses a 40 by 30 by 3.175 mm plate, nominal aluminum,
an 8 mm-radius / 1.5 mm-deep backside recess, a normal unit-total-force
100 kHz two-cycle Hann burst, and opposite-corner disk measurement. Source
and receiver covectors are checked again against native shapes at their
actual positions. The source is a mechanical test load; its relation to
real clamped PZT voltage and in-plane/out-of-plane traction remains open.

All five requested 31.5 us intervals complete, totaling 48,384 internal steps:

| Control | Internal steps | Internal step, ns |
| --- | ---: | ---: |
| Intact nominal | 8,064 | 3.90625 |
| Intact reciprocal | 8,064 | 3.90625 |
| Intact time-fine | 16,128 | 1.953125 |
| Intact remounted source | 8,064 | 3.90625 |
| Recess, matched mounts | 8,064 | 3.90625 |

The coarse reference stores 127 points at 0.25 us spacing; the fine reference
stores 253 at 0.125 us. Both contain the scenario's 64 nominal 0.5 us sample
times. Internal integration and recording intervals remain separate.
Reciprocity discrepancy is 2.67696e-15 relative. Halving the time step changes
the trace by 3.34041e-6 relative (0.000334041%), below the prospective 0.1%
limit. Post-burst modified-energy relative range is below 6.81e-16, final
GPU/CPU force discrepancy below 2.19e-13, and independent final modified-energy
reconstruction agrees within 4.45e-16 relative.

The combined source shift/contact-radius change produces 23.4409% waveform
L2 change in this fixture. Its matched-mount recess contrast is 206.326%.
These numbers quantify this small discrete control's sensitivity. Wave
spatial convergence, experimental mounting repeatability and defect-detection
performance require separate evidence. This fixture differs in geometry,
discretization and receiver selection from the earlier P1 mounting example.

## Verification, resources and use

The final focused integration passes 23 tests without skips in 18.853 test
seconds / 20.178 monitored seconds. It covers represented polynomial moments
on the affine central face, curved crossings/vertices, edge-tangent disks,
force/torque balance, intact/recess top-trace agreement, actual remounts,
unit-port gain separation, invalid geometry/state/allocation rejection,
mapped GPU operators and existing native mounting pulses. The earlier
five-test pass is retained. Overlapping groups are not added to older totals.

Native and pulse jobs peak at 546.55 MiB combined process-tree RSS, 54.0%
system RAM, minimum available RAM 7,464.56 MiB and observed GPU use
1,076 MiB / 58 C. Final tests observe 466.79 MiB RSS and GPU use 1,074 MiB /
55 C. GPU readings include desktop use; CPU temperature is unavailable.
The independent audit recomputes vector errors, force/first-moment balance,
time/reciprocity/sensitivity comparisons, post-burst energy ranges, final
CPU/GPU discrepancies and final energy. Its PNG/PDF was visually checked.

```powershell
.\.venv\Scripts\python.exe -m examples.mapped_patch_reference --order 3 --output outputs/new-port-reference
.\.venv\Scripts\python.exe -m examples.mapped_wave_reference --output outputs/new-mapped-pulse
```

Both commands require fresh project output directories, freeze source/runtime
and retain guarded workers, plans, snapshots, proof arrays and receipts. Raw
evidence is `outputs/mapped-port-{recess,intact}-q*-20261111`,
`outputs/mapped-pulse-20261111` and `outputs/mapped-port-audit-20261111`.
Compact evidence is `evidence/mapped-surface-pulse-debug.json`, fingerprint
a9f0051ccc206fddc6f3493e9591c35ceb316d82e1cbf6e7adb43ffbc043bdb4.

Full-size mapped capacity and space/time/frequency convergence, identified
PZT/clamp traction and contact/electronics response, physical validation,
accepted waveform banks and meaningful matched model/policy comparisons
remain required. No training or accepted research data generation ran.
All new references retain `ready_for_training_data=false`.
