# Complete-window temporal refinement

This study keeps the 500 x 500 x 3 mm intact mesh, nominal material, 100 kHz
two-cycle normal force disk, two receiver disks and complete 500-microsecond
window fixed. Only the explicit increment changes. It reuses the retained
31,469-step run and declares 62,938 / 125,876 steps for the finer levels.
The configuration and criteria were committed before the fine measurements.

`configs/enriched-time-refinement.json` fixes the coarse result fingerprint,
three run locations, duration and expected step counts. The comparison uses
raw per-receiver discrete L2 norms over every coarse time sample. It requires
successive-difference ratios in [3.8,4.2] and a finest relative change below
1%. Changes must exceed 1e-11 of the receiver norm to identify a rate. Missing,
shifted, partial, nonfinite or zero/unidentifiable records cannot establish
the declared second-order convergence. Receiver norms are separate, so a large
signal cannot hide a weak receiver's incorrect temporal order.

The quarter-level CFL fraction is 0.1999976931808441, derived from the retained
coarse step count. A literal 0.2 would round to 125,875 rather than 125,876
steps. This small adjustment gives exact quarter increments at the same
physical endpoint; the comparator uses subsampling and no interpolation.
The half-level fraction 0.4 already gives exactly twice the coarse count.

## Runs and assessment

Run one GPU job at a time, from the project directory, with fresh outputs:

```powershell
.\.venv\Scripts\python.exe -m examples.enriched_gpu_wave --capacity outputs/matrixfree-fullplate-capacity-20261107 --settings configs/enriched-gpu-wave-half-dt.json --mode wave --output outputs/enriched-gpu-fullplate-half-dt-20261108 --max-seconds 1800
.\.venv\Scripts\python.exe -m examples.enriched_gpu_wave --capacity outputs/matrixfree-fullplate-capacity-20261107 --settings configs/enriched-gpu-wave-quarter-dt.json --mode wave --output outputs/enriched-gpu-fullplate-quarter-dt-20261108 --max-seconds 3000
.\.venv\Scripts\python.exe -m examples.enriched_time_refinement --output outputs/enriched-time-refinement
```

The longer wall caps accommodate increased step counts and the existing
driver's conservative short-action timing preflight. The actual memory and
temperature guards stay unchanged: 1 GiB combined process RSS, 85% system
RAM, 2 GiB available RAM, 90% VRAM and 80 C GPU temperature. CPU temperature
is unavailable. No model or policy training is involved.

The assessment checks complete receipts, raw archive hashes and postburst
energy. All numerical sources, forward drivers, runtime/GPU packages and mesh
input identities must match. Source/receiver settings may not change. The
native coefficient map, coordinates, positive mass and every load/probe
coefficient must agree exactly across levels. Device identity/precision and
the mesh-specific stability bound are fixed. Actual receiver time arrays
must cover the complete window with exactly matched coarse samples.

Five new analysis tests plus two existing waveform-preflight tests pass
without skips. They exercise known first/second-order records, weak-receiver
isolation, shifted/missing samples, unidentifiable rates, changed physics/code
and tampered archives. This selected seven-test result is not added to the
earlier complete 268-test count. The simulation implementation is unchanged.

## Half-step measurement and independent CPU check

The 62,938-step half case completes all 500 microseconds in 420.403 propagation
seconds / 470.163 monitored seconds. Raw differences from the coarse trace are
0.0574956% and 0.0382134% per receiver. Its native map, coordinates, positive
mass and physical footprint coefficients agree exactly with the coarse case.
The postburst modified-energy relative range is 6.47e-16. This two-level
comparison alone does not identify temporal order.

`examples.verify_enriched_window` independently reconstructs mass and the
final CPU stiffness action from geometry. It checks saved-state energies and
receiver covectors without a CUDA allocation. The half case's mass agrees
exactly; final energy errors are 3.11e-14 relative and the final probe check
agrees exactly in this saved result. Two tests exercise arbitrary native
permutations against direct assembled energies and invalid map/state rejection.

```powershell
.\.venv\Scripts\python.exe -m examples.verify_enriched_window --run outputs/enriched-gpu-fullplate-half-dt-20261108 --steps 62938 --output outputs/enriched-half-cpu-check
```

## Scheduling during longer checks

The half case peaks at 79 C on the GPU, near the unchanged 80 C stop. The
quarter case uses `examples.pace_gpu_worker`, limited to one live leaf worker
whose module and explicit project plan path match. It pauses host submission
at 78 C and resumes at 72 C or cooler, retaining the CUDA state and numerical
step. Force and observation times depend on the step count, not wall time.
The original monitor keeps its memory, temperature and total-time guards.
No other application's process or global hardware setting is changed.

The helper records the verified PID/create time, plan/source hashes,
temperatures, scheduling events and pause duration. Cleanup resumes its owned
worker if needed. Three mock-process tests cover hysteresis, cleanup, a failed
suspend and an already-exited worker. Paused wall time counts in reported
propagation seconds; these timings are not an unpaced throughput comparison.
Pacing records are under `outputs/enriched-quarter-dt-pacing-20261108`.

The quarter measurement is in progress at this commit. A passing time result will
still not establish spatial/thickness convergence, recess scattering, PZT
voltage accuracy or physical agreement. `ready_for_training_data` stays false.
