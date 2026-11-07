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
.\.venv\Scripts\python.exe -m examples.guarded_highorder_reference --plan outputs/waveform-control-plan.json --settings configs/highorder-directional-p3-refinement.json --output outputs/highorder-directional --monitor-output outputs/highorder-directional-monitor --cache-dir outputs/highorder-reference-groups
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

The original P4 configuration (`highorder-directional-refinement.json`) keeps
degree four fixed across the following seven cases. The reproduce command
above selects the finer P3 study described below instead.

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

## Completed P4 study, November 6, 2026

All seven cases completed with the original geometry, source, settings and
tolerances. The complete study returned numerical-screen status 2.

| Axis | First pair, worst relative L2 | Final pair, worst relative L2 | Fixed limit | Screen |
| --- | --- | --- | --- | --- |
| In-plane cells 3 / 4 / 5 | 58.859% | 18.888% | 5% | Failed |
| Thickness cells 1 / 2 / 3 | 1.7563% | 0.20932% | 5% | Passed |
| Internal step 125 / 62.5 / 31.25 ns | 0.88688% | 0.22166% | 1% | Passed |

The temporal difference ratio gives order 1.99886. Every receiver exceeds the
declared RMS floor. This identifies in-plane resolution as the remaining
failed axis for these discretizations; it does not establish a continuum
error estimate or remove physical/PZT validation requirements.

The outer monitor recorded 305.48 seconds, 628.5 MiB peak combined RSS,
72.6% peak system RAM and at least 4,450.7 MiB available. GPU observations
peaked at 995/8,192 MiB and 40 C; calculation used CPU only. CPU temperature
was unavailable. No training ran. Both raw and difference plots were checked.

A new-output recovery completed in 7.20 seconds inside the study runner,
with seven cache hits, zero new solves and bitwise-identical displacement
arrays. It retained the same failed-screen status. Raw artifacts remain in
`outputs/highorder-directional-20261106`, with recovery in
`outputs/highorder-directional-recovered-20261106`.

## Finer P3 study, November 6, 2026

The existing `highorder-directional-p3-refinement.json` changes degree and
in-plane cell counts while keeping the same physical problem and fixed
acceptance limits. It compares 7/8/9 in-plane cells, 1/2/3 thickness cells
and 125/62.5/31.25 ns steps. The finest 9 x 9 x 3 case has 23,520 vector
coefficients. This is a separate study, not a change to the failed P4 result.

| Axis | First pair, worst relative L2 | Final pair, worst relative L2 | Fixed limit | Screen |
| --- | --- | --- | --- | --- |
| In-plane cells 7 / 8 / 9 | 6.7483% | 3.4578% | 5% | Passed |
| Thickness cells 1 / 2 / 3 | 5.1377% | 0.59389% | 5% | Passed |
| Internal step 125 / 62.5 / 31.25 ns | 0.89633% | 0.22396% | 1% | Passed |

All seven cases completed and the study returned 0. The observed temporal
order is 1.99907. The maximum linear residual is 2.07e-15 and maximum
postburst relative energy range 3.90e-14. A supplemental comparison of the
finest P4 and P3 traces differs by at most 2.1195%; this diagnostic has no
additional acceptance threshold and is not a continuum error bound.

The outer monitor recorded 408.61 seconds, 684.4 MiB peak combined RSS,
73.7% peak system RAM and at least 4,262.0 MiB available. GPU observations
peaked at 1,007/8,192 MiB and 45 C; calculations used CPU. CPU temperature
was unavailable. Raw and difference PNG/SVG figures were inspected. The P3
artifacts are in `outputs/highorder-p3-directional-20261106`.

All 213 regression tests passed with real Elmer/NGSolve checks and no skips.
The monitored run took 210.31 seconds at 771.0 MiB combined RSS, 73.7% system
RAM and 45 C GPU temperature. A follow-up of five resource/child-termination
checks passed after directing the wrapper's temporary files into the project.
Dependency checks found no broken requirements. No full training ran.

The compact record is [highorder-waveform-refinement-debug.json](evidence/highorder-waveform-refinement-debug.json).
It binds source/runtime identities, per-case worker checks, comparison arrays,
monitor receipts, plot hashes, cache recovery and test results. The small
intact mechanical control passes its declared discrepancy screen. Full-size
plates, recess scattering, independent continuum agreement, bonded-PZT voltage
responses and physical validation remain open; training readiness stays false.
