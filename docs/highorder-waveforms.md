# Higher-order intact-plate waveform refinement

The optional NGSolve mechanical reference can now propagate one bounded
transmitter/frequency group in an exact intact box. A predeclared study
separates in-plane, through-thickness and time refinement. Every result
retains `ready_for_training_data=false`: this is not yet a validated
research-plate or bonded-PZT simulation.

## Reproduce

Install the optional project-local dependencies described in
[highorder-reference.md](highorder-reference.md). Use an existing noise-free
scenario plan. The study below uses the same 40 x 40 x 3 mm control as the
earlier P2 work, rather than changing the physical problem to suit the
higher-order solver.

```powershell
.\.venv\Scripts\python.exe -m examples.guarded_highorder_reference --plan outputs/waveform-control-plan.json --settings configs/highorder-directional-refinement.json --output outputs/highorder-directional --monitor-output outputs/highorder-directional-monitor --cache-dir outputs/highorder-reference-groups
.\.venv\Scripts\python.exe -m traxis plot-directional-waveforms --sweep outputs/highorder-directional --records 0 3 --output outputs/highorder-directional-figures
```

Use new output/monitor directories on each attempt. The outer wrapper
defaults to 600 seconds, 1,024 MiB sampled combined RSS, system RAM below
85% and at least 2,048 MiB available. It monitors the whole observed process
tree and terminates it if a limit is exceeded. Native work uses one CPU
thread; no GPU arrays are allocated. GPU memory/temperature are observed
system-wide where supported. Missing CPU temperature readings remain null.

The underlying command is `check-highorder-waveforms`. Completed numerical
screen failures return 2, operational failures return 1 and successful
discrepancy screens return 0. None promotes data for research training.

## Physical problem and independent axes

The intact control has eight equivalent circular footprints of radius
3 mm, inset 6 mm. TX 0 drives a 1 N normal-force, two-cycle 100 kHz Hann
burst; its seven eligible receivers measure area-averaged normal
displacement. There are 128 output samples at 0.5 microseconds. The solid
is undamped, isotropic and traction-free. Apertures are analytic circles,
not vertex or element-centroid selections. This does not specify PZT,
adhesive, electrodes or acquisition electronics.

The seven definitions keep degree four fixed:

| Case | Cells x/y/z | Vector coefficients | Internal dt | Steps |
| --- | --- | --- | --- | --- |
| 0 | 3 / 3 / 3 | 6,591 | 31.25 ns | 2,032 |
| 1 | 4 / 4 / 3 | 11,271 | 31.25 ns | 2,032 |
| 2 | 5 / 5 / 3 | 17,199 | 31.25 ns | 2,032 |
| 3 | 5 / 5 / 1 | 6,615 | 31.25 ns | 2,032 |
| 4 | 5 / 5 / 2 | 11,907 | 31.25 ns | 2,032 |
| 5 | 5 / 5 / 3 | 17,199 | 125 ns | 508 |
| 6 | 5 / 5 / 3 | 17,199 | 62.5 ns | 1,016 |

Cases 0/1/2 change only in-plane spacing, cases 3/4/2 change only thickness
spacing, and cases 5/6/2 change only the time step. The temporal comparison
requires identical saved mesh, aperture covectors and operators. Every
case uses the same declared source, native runtime, material, physical
footprints, receive actions and output grid.

The fixed last-pair tolerances are 5% in-plane, 5% thickness and 1% time
relative L2 error at the worst receiver. All records must exceed a 1e-18 m
reference RMS floor. The comparisons retain boundary reflections and rigid
motion and use float64 displacements without amplitude fitting, shifting,
alignment or resampling. Envelope and carrier-phase diagnostics supplement
the raw errors; they are not verified modal arrival times.

The time sequence is geometric and reports its difference-ratio estimate
of observed order. Thickness levels 1/2/3 are not geometric, so the report
leaves their observed order null. No Richardson extrapolation or continuum
error bound is claimed from these differences.

## Resource screening and provenance

Complete grid counts, polynomial degree and conservative assembly memory
are screened before native allocation. The P1-P4 operators and aperture
interfaces have separate exact-field and independent Elmer checks, linked
from [highorder-apertures.md](highorder-apertures.md). Each waveform case
also checks its actual covered disk areas and every represented centered
monomial before propagation, box volume, solve residuals and postburst
energy retention.

Assembly and Newmark factorization/time integration run in a monitored
worker. Meshes, covectors, sparse stiffness/consistent mass, energy history
and raw probes are saved without pickle. Plans bind the complete Python
source snapshot and native package/library identities. Parent-side checks
verify every saved native artifact and the exact output arrays/time grid
before comparing results.

The cache uses fingerprinted physical jobs and retains separate attempts.
Only a completed, checked worker publishes a receipt. A cache hit rechecks
the job, worker report and all artifact hashes; changed or incomplete
evidence cannot silently count as a solve. Failed attempts are preserved.
Changing source, native libraries, geometry, degree, time step, acquisition
or declared scenario changes the key.

A 6 x 6 x 3 P4 cost probe exceeded the existing four-million-entry export
limit before time integration. The following 5 x 5 x 3 probe completed in
19.44 seconds at 477.8 MiB sampled combined RSS, with 3,394,803 stiffness
entries and 1,148,049 mass entries. This selected a feasible study size;
it was not a convergence result. Its failure/success receipts are retained.
