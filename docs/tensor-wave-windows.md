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
November 10. The other four cases remain required before convergence can
be assessed.

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

Research-plate tensor convergence, representative recesses and frequencies,
calibrated PZT/electronics and physical agreement remain open.
The previous failed tetrahedral thickness screen is preserved. No research
bank or full model/policy training is accepted by these preflight checks.
