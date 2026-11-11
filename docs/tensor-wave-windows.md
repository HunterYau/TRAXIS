# Complete tensor mechanical wave windows

The wave runner retains the intact 500-by-500-by-3 mm mechanical experiment:
100 kHz two-cycle Hann excitation, 1 N force scale, a source at (20,20) mm,
receivers at (250,250), (480,480) and (350,200) mm, and 3 mm radius top disks.
`tensor_patches.py` integrates the same physical areas at each polynomial
degree. The recording interval is 500 microseconds. These are mechanical
normal-force/area-average-displacement ports, without calibrated PZT voltage.

`examples.tensor_wave_study prepare` checks the fixed protocol and freezes
five unique runs before measurement. All use 150-by-150-by-1 cells. The
spatial comparison varies Q3/Q4/Q5 at 65,536 time steps; the time comparison
uses Q4 at 32,768/65,536/131,072 steps. The middle run is shared. Spatial
differences must decrease and the finest raw change must be below 5% at
every receiver. Temporal difference ratios must be 3.8-4.2 and finest changes
below 1%. Raw comparisons cover the entire interval, with no phase fitting,
envelope replacement or receiver/window changes. p-refinement changes the
polynomial resolution in every axis; it does not isolate thickness-cell error.

Plans bind the complete core source inventory, four drivers, native/GPU
runtime, physical settings and output locations. Source snapshots are saved.
Every run first checks an actual full-size CUDA action against streamed CPU
action and positive box mass moments, then propagates the requested interval
under a process-tree guard. Sparse footprint coefficients and raw traces,
energies, final states and mass are retained. The reader checks hashes,
fingerprints, exact time grids and shape-derived ZIP/NPY bounds before loading
arrays. Interrupted or incomplete windows cannot enter a comparison.

The independent CPU checker assembles mass by bounded local-element scatter,
separate from the production axis-factorized construction. It recomputes the
saved state's stiffness action, physical/modified energies and circular
probe readings. Each comparison requires a matching successful CPU receipt.
These are saved-state checks, not an independent CPU trajectory replay.

## Preflight evidence

The current-source footprint/native/CUDA/cooling group passes all 24 tests
without skips in 15.153 test / 16.489 monitored seconds. The separate protocol,
comparison, archive-bound and saved-state math group passes nine tests in
0.169 test / 1.439 monitored seconds. Groups overlap; they do not supersede
the last complete integrated 268-test result.

An end-to-end 40-by-30-by-3 mm smoke fixture completes all 32,768 steps across
500 microseconds, using 1,215 vector coefficients. It takes 13.235 propagation
seconds. CUDA/CPU action discrepancy is 3.40e-16, modified-energy range is
3.66e-15, and independent final energies agree within 2.23e-16. Peak process
RSS is 438.26 MiB; system RAM peaks at 49.9%, available RAM stays above
8,127.19 MiB, and GPU observations reach 1,047 MiB / 53 C. CPU temperature is
unavailable. This checks the workflow, not research-plate accuracy.

Raw directories are `outputs/tensor-footprint-preflight-20261109`,
`outputs/tensor-wave-protocol-tests-20261109`,
`outputs/tensor-wave-smoke-20261109` and
`outputs/tensor-wave-smoke-cpu-20261109`. Compact evidence is
`evidence/tensor-wave-preflight-debug.json`, fingerprint
`10dd64874f45992af7a655415e64d2ffb75e3d9fa6d57cc7f4e2cf9ebd26d967`.
The retained collector initially lacked the project import root when invoked
as a standalone script; that collector-only invocation was corrected before
writing evidence. Simulation source and numerical results were unchanged.

## Reproduction

Use a new directory for each run and retain the source version once a study
is prepared. Do not edit core sources or the bound drivers between cases.

```powershell
.\.venv\Scripts\python.exe -m examples.tensor_wave_study prepare --output outputs/tensor-study
.\.venv\Scripts\python.exe -m examples.tensor_wave --study outputs/tensor-study/tensor-study-plan.json --case Q4-N32768 --max-seconds 1800
.\.venv\Scripts\python.exe -m examples.verify_tensor_window --run outputs/tensor-study/runs/Q4-N32768 --output outputs/tensor-study/cpu/Q4-N32768
```

Cooling control can attach to the exact live worker with
`examples.pace_gpu_worker --module examples.tensor_wave --plan <run>/tensor-wave-plan.json --output <new-pacing-directory>`.
It suspends host submission at 78 C and resumes at 72 C, while the original
80 C guard and numerical state remain active. It never selects an unrelated
process. After every required case and CPU check completes, run
`examples.tensor_wave_study spatial` or `time` with `--plan` and a new
`--output`. A configuration or small smoke result is not a completed study.

## First completed research-plate window

The frozen study is `outputs/tensor-fullplate-study-20261109`, fingerprint
`c339dd16680f52c9de65b3507bf5511b6d115228fc993b4afa48df6bcfbd5505`.
Its Q4-N32768 case completes the actual 500 mm plate / 500-microsecond
window with 5,418,015 vector coefficients. The step is 15.2587890625 ns,
below the 37.8375 ns conservative stability bound. The worker completed on
November 9; its saved-state CPU audit and final figures were checked on
November 10. At that cutoff the other four cases remained required. The
subsequent half-step measurement is recorded below; three cases remain.

| Measurement | Observed result |
| --- | ---: |
| Propagation / monitored worker time | 243.828 / 259.031 s |
| Full CUDA / independent CPU action discrepancy | 9.87e-16 relative |
| Postburst modified-energy range | 2.33e-15 relative |
| CPU final physical and modified energy agreement | 3.34e-16 relative |
| CPU independently scattered mass agreement | 4.14e-16 relative |
| Final area-average probe agreement | 2.44e-16 relative |
| Peak combined RSS | 919.96 MiB |
| Peak system RAM / minimum available RAM | 53.1% / 7,607.75 MiB |
| System GPU memory / temperature peak | 1,551 MiB / 74 C |

All four circular covectors reproduce their saved coefficients exactly;
analytic area and represented moment errors are below 5.45e-15. Receiver
peak magnitudes are 0.517158, 0.907062 and 0.456410 nm. The cooling monitor
finishes normally without a suspension. CPU temperature remains unavailable.
Raw source-bound states, probes, operator fields and resource receipts are
retained in the study. The independent CPU receipt is under `cpu/Q4-N32768`.

`outputs/tensor-fullplate-q4-window-20261110-v2` contains audited PNG/PDF
receiver and energy/resource figures. Both were visually checked. Resource
plots use separate elapsed clocks for the parent monitor and cooling monitor;
the first draft overlay was corrected without changing any numerical data.
Compact evidence is `evidence/tensor-fullplate-window-debug.json`, fingerprint
`faf6f1bce170be69fc3fa02d14a4bff058709750caf0f53de28812b4773fd0c3`.
The collector is `outputs/tensor-wave-tools-20261109/collect_window.py`.

## Production-partition periodic profile check

An additional frozen 72-case CPU study checks Q3/Q4/Q5 at the actual
3.333333 mm lateral cell width and one 3 mm thickness cell. It retains all
twelve 5-250 kHz frequency samples, A0/S0 analytic profiles, 99% profile mass
coverage and the unchanged 0.05% worst-selected-frequency design criterion.
Mode selection uses profile overlap without consulting eigenfrequency.
All 72 generalized eigenproblems, selected vectors and reference profiles
are retained and independently re-solved by a separate auditor.

| Degree | Passing cases / 24 | Worst support frequency error |
| --- | ---: | ---: |
| Q3, coarse comparison | 16 / 24 | 0.697583% |
| Q4, required candidate | 24 / 24 | 0.00730436% |
| Q5, required candidate | 24 / 24 | 0.000261640% |

Q3's eight failures remain in the record. The frozen required candidates
are Q4 and Q5; this does not relabel Q3 as accurate. The CPU worker completes
in 2.336 monitored seconds at 239.59 MiB RSS. Maximum saved selected-eigenpair
residual is 6.24e-10; independently recomputed modal overlaps agree exactly.
Raw data are `outputs/tensor-production-profile-20261110-v2`; the producer and
auditor are in `outputs/tensor-wave-tools-20261109`. Compact evidence is
`evidence/tensor-production-profile-debug.json`, fingerprint
`f1fc599615893f3498a5fffb7945fdf30994c9027c24e4a173feb36f75ff5467`.
The initial analysis bound the core/wave drivers but omitted the two imported
analysis-driver identities. They and the producer were explicitly bound and
archived, then all 72 cases were repeated under identical criteria and gave
identical numerical results. The original raw receipt and producer snapshot
remain in the directory without the v2 suffix.
GPU observations in that CPU receipt include the concurrent independent
Q4 wave job. This is an intact uniform periodic screen, not recess scattering,
finite-plate convergence or a PZT/electronics transfer-function calibration.

## Completed half-step window and first raw temporal difference

Q4-N65536 completes the same 500-microsecond window at 7.62939453125 ns.
Propagation takes 494.814 seconds; monitored worker time is 510.183 seconds.
All 65,536 steps and independent CPU mass, energies and receiver checks pass.
Both recomputed final energies agree at the recorded floating-point precision;
mass agrees within 4.14e-16 and final probes within 1.63e-16 relative.
Modified-energy range is 2.33e-15. Physical disk coefficients reproduce
exactly. Peak combined RSS is 920.70 MiB, system RAM 57.5%, minimum available
RAM 6,901.21 MiB and GPU observations 1,449 MiB / 75 C. The cooling monitor
finishes without a suspension. CPU temperature remains unavailable.

The two levels have exactly matching mass, circular covectors, source/runtime
identities and common coarse time samples. Their complete raw receiver L2
changes, relative to the finer trace on those samples, are:

| Receiver | Raw relative change | Maximum absolute difference |
| --- | ---: | ---: |
| (250,250) mm | 0.0549479% | 0.401280 pm |
| (480,480) mm | 0.0364132% | 0.462014 pm |
| (350,200) mm | 0.0504256% | 0.276288 pm |

This is a two-level diagnostic, with `convergence_assessed=false`; no rate or
three-level temporal pass is claimed. Q4-N131072 and spatial Q3/Q5-N65536
remain required. Raw half-step data and CPU/pacing receipts are in the frozen
study. Audited PNG/PDF figures and matched traces are in
`outputs/tensor-fullplate-q4-half-window-20261110` and
`outputs/tensor-first-time-pair-20261110`; all three new figures were visually
checked. The compact receipts are `evidence/tensor-half-window-debug.json`
(fingerprint `eb4cfcc4790e28f354c1fe173c2a47742a0839893dfcb93d55d653dcc87b060b`)
and `evidence/tensor-first-time-pair-debug.json`
(`0e3b7501206656ee421c81e99c5334bc12edec4b855168ff4e7a13eab2dd2ae5`).

## Completed Q3 spatial level

Q3-N65536 completes the fixed 500-microsecond experiment with 2,440,812
vector coefficients and the common spatial step of 7.62939453125 ns.
Propagation takes 224.531 seconds and the monitored worker 233.804 seconds.
Full GPU/CPU stiffness agreement is 4.50e-16 relative. Independent local-scatter
mass, final energies and probe checks agree within 3.96e-16. Postburst
modified-energy range is 1.72e-15. All physical patch coefficients reproduce
exactly; analytic area/moment errors remain below 1.56e-14.

Peak combined RSS is 634.79 MiB, system RAM 51.1%, minimum available RAM
7,934.41 MiB and system GPU observations 1,185 MiB / 74 C. Cooling finishes
without a suspension; CPU temperature remains unavailable. Waveform and
energy/resource figures in `outputs/tensor-fullplate-q3-window-20261110`
were visually checked. Raw states and CPU/pacing receipts remain under
Q3-N65536 in the frozen study. Compact evidence is
`evidence/tensor-q3-window-debug.json`, fingerprint
`4114a4cdc5b96637e63626e1b15bebf20782a620f5950513984c6b4c5081debf`.
This verifies completion and arithmetic for the declared coarse spatial
level; Q3's eight periodic-band failures remain in the earlier record.
Q5-N65536 and Q4-N131072 are still required before the full assessment.

## Completed Q5 window and passing full-window spatial screen

Q5-N65536 completes all 65,536 steps across the fixed 500-microsecond
window with 10,152,018 vector coefficients. Propagation takes 973.007 seconds
and the monitored worker 998.247 seconds. Full CUDA/CPU action discrepancy
is 1.06e-15 relative. Independent CPU mass, final energies and probes agree
within 4.19e-16. Raw modified-energy range is 1.24e-15. The four physical
covectors reproduce exactly, with analytic area/moment errors below 1.35e-14.
Peak RSS is 1,370.93 MiB, system RAM 56.6%, minimum available RAM
7,043.56 MiB and GPU observations 1,881 MiB / 77 C. CPU temperature is
unavailable. The 3,600-second wall cap allows this measured higher-degree
cost; physical settings and accuracy criteria are unchanged.

The optional cooling monitor stops after 744.828 seconds when Windows denies
atomic replacement of its progress log. The simulation and its separate
resource guards continue. An unchanged monitor reattaches to the same PID
and creation identity, completes a 29.782-second segment and records no
suspension. Both segments and the error remain retained. The monitoring gap
cannot be inferred exactly from their separate relative clocks. The original
parent/worker observations cover the full job and peak at 77 C. The window
is neither restarted nor shortened. No numerical source or physical input
changes during this recovery.

The Q3/Q4/Q5 raw full-window comparison now passes its frozen spatial screen.
All independent saved-state CPU checks are required and pass. Differences
decrease at each receiver; Q4-to-Q5 changes are:

| Receiver | Finest raw relative change | Coarse/fine difference ratio |
| --- | ---: | ---: |
| (250,250) mm | 0.117856% | 193.173 |
| (480,480) mm | 0.100770% | 149.842 |
| (350,200) mm | 0.109101% | 190.093 |

All are below the unchanged 5% criterion. These are p-refinement differences
for the intact mechanical experiment, not a continuum-error bound, a
fixed-order h-convergence rate or acceptance of another defect/frequency.
The earlier failed tetrahedral thickness study and Q3 mode failures remain
historical results, not retroactively passing checks.

The independent auditor recomputes every raw metric and decision, verifies
the exact retained comparison arrays and source/runtime/device identities,
and preserves CSV plus PNG/PDF artifacts. Its final figures were visually
checked. Raw assessment and audit directories are
`outputs/tensor-spatial-assessment-20261110` and
`outputs/tensor-spatial-audit-20261110-v2`. The initial plot placed a legend
over a bar; the final layout moves it below the plot, preserving the initial
artifact and leaving all numerical metrics unchanged.

Q5 waveform and energy/resource figures are in
`outputs/tensor-fullplate-q5-window-20261110`, with separate panels for both
cooling segments. Compact receipts are `evidence/tensor-q5-window-debug.json`
(fingerprint `c97c6850952793e6cdac9891030a5783165a34ea9ced0620caee403ee09318ca`)
and `evidence/tensor-spatial-refinement-debug.json`
(`639e61c6cf2915233a33641a7c5c3ae66bf74f22f9af4177eb0d1bcda59cff21`).
The final Q4-N131072 temporal level is still required at this cutoff.

Tensor temporal convergence, representative recesses and frequencies,
calibrated PZT/electronics and physical agreement remain open.
The previous failed tetrahedral thickness screen is preserved. No research
bank or full model/policy training is accepted by these preflight checks.
