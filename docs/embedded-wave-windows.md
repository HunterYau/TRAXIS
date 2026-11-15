# Wave windows on the embedded recess grid

The new wave runner advances the explicit curved mesh through a declared
mechanical observation window. It uses the checked float64 CUDA stiffness,
positive GLL mass and device-resident velocity Verlet. A separate CPU
readback rebuilds the geometry/operator and checks saved results. No model
training or accepted waveform-bank generation is part of this workflow.

## Cases and frozen inputs

Both cases use explicit 3.175 mm stock, cubic geometry, one layer on each
side of the recess-floor partition, a 100 kHz two-cycle Hann-windowed
sinusoid and a unit peak normal force distributed over a 3 mm-radius disk.
The receive observable is area-average normal displacement in meters.
These are mechanical controls, not PZT voltage predictions.

The small case is 80 by 60 mm, with 2 angular cells, radial counts (1,1),
40 mm exterior spacing and a 31.5 us window. It uses 8,064 or 16,128 steps
at Q4. Source center is (6,6) mm and receivers are (40,30), (74,54) and
(60,15) mm. Its partitioning recess is centered at (41,29) mm with radius
8 mm and depth 1.5 mm. Its deliberately coarse geometry is a software fixture.

The full case is 500 by 500 mm, with 12 angular cells, radial counts (2,6),
5 mm exterior spacing and a 500 us window. The nominal recess is centered
at (255,245) mm with radius 15 mm and depth 1.5 mm. Source center is (20,20)
mm and receivers are (250,250), (480,480) and (350,200) mm. Both intact and
recess states are supported with the same partition. A case is rejected if
its step exceeds 80% of the conservative stability bound.

Plans seal all physical/numerical controls, predicted mesh counts, core and
driver hashes, CPU/native/CUDA package identities and a source snapshot.
The runner checks those identities before and after propagation. Incomplete
or failed runs retain their log/report and cannot become a completed window.
The requested interval is never shortened to meet a resource limit.

## Guards and cooling

The current full worker allows 3,600 seconds and 4,096 MiB process-tree RSS;
the small worker allows 180 seconds and 1,024 MiB. The extended launcher keeps
a 2,048 MiB process cap at Q3/Q4 and allows 4,096 MiB at Q5. System RAM stops
at 89.9% used, with at least 768 MiB available, following the user's approved
90% ceiling. The GPU-memory limit remains 90%, the GPU temperature cap 80 C,
and the CPU cap 90 C when a supported sensor exists. CPU temperature is
unavailable on this host and remains unverified. Earlier runs retain their
original resource profiles; their receipts are never rewritten.

Host submissions pause between synchronized GPU batches at 76 C and resume
after the observed temperature reaches 72 C or lower. The physical state,
time step and step count do not change. Guard checks continue throughout
cooling. A missing single-GPU temperature sensor fails visibly. Every pause
and its duration is retained; no power, fan, clock or operating-system setting
is changed. Process-tree monitoring also enforces limits from the parent.

The measured coarse window projects about 2,573 seconds (43 minutes) for
the 294,912-step case, including the observed setup and cooling overhead.
`examples.embedded_wave_budget` therefore supplies an explicit 3,600-second
execution profile. It calls the same frozen numerical worker and records
its own source plus every actual limit in the plan. Physical-case and
numerical acceptance limits retain their original values. Current memory
limits follow the authorized ceiling described above. The original standard
plans keep their original 2,400-second and 2,048 MiB budgets.
Comparisons check both parent and worker limits against the appropriate
declared profile; prior source identities are preserved.

The extended guard compacts observation history before 8,000 entries,
retaining first/latest samples, a regular older sample sequence and every
resource extremum. It records the total observations and compaction count.
This keeps late readings and fresh process-tree samples available beyond
the base guard's 10,000-entry history capacity. Actual checks, enforcement
and cooling continue at their original cadence. Synthetic long histories
test peak retention, fresh mutable samples and visible limit failures.

The base guard now also keeps a separate fresh `latest_sample`, even after
its retained history fills. The external-process monitor augments that
sample, preventing repeated addition of child RSS into a stale history
entry. A regression exercises 10,002 observations. The extended launcher
continues compacting history to retain late extrema as well as fresh checks.

## Saved arrays and independent checks

Each window saves every trace sample, physical and modified energy samples,
final displacement/velocity, final GPU force and the mass diagonal. Ports
retain signed weights, node indices and direction. Readers check array
headers and expansion bounds before loading, exact declared time grids,
state dimensions, finiteness, positive mass, port topology/total force,
completed-step statistics, receiver identifiability and postburst energy.
Artifact hashes bind the raw arrays to parent and child reports.

`examples.verify_embedded_window` rebuilds the CPU system from the frozen
case. It independently checks final stiffness action, mass, physical and
modified energy, force/first moments on all four physical ports and the
final sparse probe contraction. This verifies the saved end state and
readout; full-trajectory and continuum accuracy require additional evidence.

`examples.embedded_native_pulse` separately replays a complete small Q4
window using independently assembled NGSolve local matrices retained from
the native reference. It converts independent native GLL mass, assembles a
bounded sparse global stiffness, reintegrates native disk loads/probes and
uses a displacement central-difference recurrence. The recurrence is checked
against an analytical forced oscillator for second-order displacement and
velocity errors. The native replay does not call the CUDA operator or the
production time integrator. Its complete-waveform tolerance is 1e-8 relative;
its force/state/identity limits remain explicit in the retained report.

The first completed native replay covers all 8,065 samples at all three
small-case receivers. Relative waveform discrepancies are 6.37e-13,
5.26e-13 and 7.49e-13. Final displacement/velocity discrepancies are
1.14e-11 and 2.21e-12; native mass agrees within 3.37e-15. The sparse
stiffness has 8,213,625 entries and symmetry discrepancy 1.22e-15. This
82.852-second CPU replay peaks at 350.23 MiB RSS. Concurrent full-wave GPU
activity is included in its GPU telemetry; the replay itself uses the CPU.

The first replay attempt treated the native scalar aperture vector as a
three-component vector and stopped before integration. The mapping now puts
the converted scalar covector in the declared normal component. The failed
report and original helper are retained; tolerances were not changed.

## Separate spatial and time comparisons

The fixed protocol is `configs/embedded-wave-protocol.json`:

| Comparison | Levels | Raw waveform criterion |
| --- | --- | --- |
| Displacement degree | Q3/Q4/Q5, each 196,608 steps | Identifiable, decreasing differences; finest change below 1% |
| Time step | Q4 at 131,072 / 196,608 / 294,912 steps | Identifiable, decreasing differences; observed order 1.7-2.3; finest change below 0.1% |

The time refinement factor is 1.5, so second-order differences have a ratio
of 2.25. All cases share 32,768 comparison intervals through integer
subsampling. Only arithmetic roundoff in the selected physical times is
allowed; no interpolation, amplitude fitting, phase alignment or signal
normalization is performed. All three receivers must pass.

The assessor requires passing independent CPU receipts and unchanged raw
hashes for every level, one physical case, matching device/precision and
identical discrete ports across time refinements. Small changes below the
identifiability floor do not establish an observed convergence rate. These
numerical criteria are declared engineering screens; measurement repeatability
is still needed to establish physical accuracy requirements.

## Measured checkpoint (November 12)

| Case | Coefficients | Steps | Internal step (ns) | Window (us) | Propagation wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small intact Q4 | 14,715 | 8,064 | 3.906250 | 31.5 | 3.571 |
| Full intact Q4 | 4,507,515 | 131,072 | 3.814697 | 500 | 1,031.080 |

The full worker completes in 1,284.275 seconds (parent monitor: 1,285.792
seconds), including setup, propagation and saved proof. The source/receiver
patches are ready at 244.273 seconds. Propagation includes 78 cooling pauses
totaling 161.129 seconds. These are observed wall times, replacing the earlier
operator-only forecast for this particular run; the forecast excluded these
other costs.

Postburst modified-energy range is 1.85e-15 relative. The separate full-size
CPU readback agrees in final stiffness force within 1.73e-13, mass exactly,
physical/modified energy within 5.56e-16 and probe contraction within 4.33e-16.
The full worker/monitor peaks at 1,480.73 MiB process-tree RSS, 62.1% system
RAM and 1,551 MiB GPU memory / 76 C, with at least 6,150.69 MiB available RAM.
The CPU readback peaks at 630.86 MiB RSS. CPU temperature remains unavailable.

The final integrated suite passes 25 tests without skips in 4.874 seconds
(6.134 seconds monitored wall time), including actual optional CUDA checks.
Its peak RSS is 341.00 MiB, system RAM 55.2%, minimum available RAM 7,264.26
MiB and observed GPU 905 MiB / 52 C. The earlier overlapping 19-, 12- and
6-test groups are retained, not added to this suite total. The first command
used two incorrect existing module names; its import-error report is retained.

An independent retained-array audit rechecks report/plan/array/source hashes,
every native reference element file, complete time grids, energy ranges,
CPU bindings and the native pulse differences. The final PNG figure was
visually inspected; a corresponding PDF is retained. See
[the compact receipt](evidence/embedded-wave-window-debug.json), fingerprint
3c5a704bf7abfccf30c69c18f4bc51aafbaf5972184366c7524c76d2f2cdfdff.

This checkpoint completes one full-size intact mechanical window and the
small independent native trajectory comparison. Full-size recess windows,
the declared spatial/time studies, representative thickness/frequency
coverage and physical PZT measurements remain open. No training ran.

## Paused checkpoint (November 12)

The user paused the research goal while the 196,608-step intact time level
was running. Its last logged progress is 73,392 steps / 186.646 us; the
process was stopped and has no accepted complete window or CPU readback.
The verified 131,072-step window above remains the completed reference.
The interrupted attempt has no restartable numerical state; resume with a
fresh output directory rather than accepting its partial progress.

The extended execution/profile checks pass 18 focused tests without skips
in 0.593 seconds. The 294,912-step level and full three-level assessment
remain pending. The longer profile is committed, but has not yet been used
for a completed production window. Generated raw outputs and local virtual
environments remain ignored by Git; source, protocols and compact completed
evidence are tracked.

## November 14 refinement attempt

The fresh 196,608-step attempt was automatically stopped when system RAM
reached 85.1%, with 2,417.88 MiB available. The preceding process-tree
reading was 1,224.82 MiB. Last logged progress was 112,064 steps; there is
no complete window or accepted waveform/CPU readback from this attempt.
The original memory threshold was retained. After stopping, available RAM
recovered to 4,239.04 MiB (73.9% used). This does not identify which other
processes contributed to the system-wide pressure.

The failure record and input/log hashes are retained in
`evidence/embedded-refinement-ram-stop-20261114.json`. That attempt remains
incomplete and has no restartable numerical checkpoint.

After the user authorized up to 90% system RAM use, the extended launcher was
changed to reject startup at 90% or higher and enforce an in-run 89.9% ceiling
with a 768 MiB minimum available-RAM floor. The 89.9% guard makes the stop
occur before a sample can reach the user's 90% limit. The 3,600-second wall
limit, 2,048 MiB process-tree limit, 90% dedicated-GPU-memory limit, and 80 C
GPU-temperature limit remain in force. The new thresholds are recorded in
each run plan; the failed attempt retains its original profile.

The local `.venv` launcher pointed to a deleted Python 3.12.10 base install.
Python 3.12.10 was restored as an ignored portable runtime under
`.venv/portable-runtime` and connected to the existing `.venv/Lib/site-packages`.
NumPy 2.5.4, SciPy 1.18.1, psutil 7.2.2, NGSolve 6.2.2606, and CuPy 14.2.0
loaded. PyTorch is the CPU build (2.15.0+cpu); the wave solver's CUDA path uses
CuPy. Its first import tried to create a cache outside the workspace and was
denied, so the cache was directed inside the ignored `.venv` folder. A real
CuPy array calculation then passed on the RTX 3060 Ti. No system-wide software
or files outside the project were changed. CPU temperature sensors were not
available, and the recorded NVIDIA telemetry does not separately report
shared GPU memory.

The RAM-profile change is committed as `0ec67e0` (`Honor the approved embedded
wave RAM ceiling`). Focused tests passed: 7 launcher/telemetry tests and 4
refinement-study tests. An initial unittest module-name invocation failed
because `tests` is not a Python package; discovery with `-s tests` was the
corrected invocation. No training updates were run.

## November 14 completed intact Q4 time refinement

The original coarse run (Q4, 131,072 steps) was compared with fresh medium
(196,608) and fine (294,912) runs. All used the full 500 by 500 by 3.175 mm
intact plate, the same mesh, material, source and receiver ports, 500 us
physical window, and CuPy CUDA solver. Each new window completed with zero
training updates and passed an independent CPU readback. Raw run directories
remain under ignored `outputs/` and are bound by their fingerprints and
hashes in `evidence/embedded-time-refinement-20261114.json`.

| Steps | GPU worker wall (s) | Maximum system RAM | Minimum available RAM (MiB) | Peak observed process-tree RSS (MiB) | GPU memory / temperature |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 196,608 | 1,574.036 | 63.6% | 5,905.07 | 1,481.13 | 2,415 MiB / 76 C |
| 294,912 | 2,370.252 | 63.1% | 5,985.01 | 1,480.32 | 2,417 MiB / 76 C |

The CPU readback force errors are 1.7361e-13 and 1.7178e-13; mass agrees
exactly; energy errors are at most 3.34e-16; and receiver contraction errors
are 4.29e-16 and 4.63e-16. Postburst modified-energy relative ranges are
1.6861e-15 and 2.0234e-15. GPU use reached 99-100% during sampled propagation.
The built-in thermal controller paused briefly at 76 C and resumed after
cooling. CPU temperature and shared GPU-memory use were not directly measured.

The three-level time-refinement assessment passed for all three receivers.
Observed orders were 2.00000114, 2.00000089, and 2.00000092; difference ratios
were 2.25000104, 2.25000081, and 2.25000084. Finest relative waveform changes
were 1.1561e-5, 7.6820e-6, and 1.0427e-5, each below the declared 0.001
(0.1%) limit. This supports second-order time convergence for this fixed
intact Q4 numerical case. It does not establish spatial convergence, real
plate accuracy, PZT contact fidelity, or model-training performance.

The medium/fine windows and CPU checks are reusable inputs for the remaining
spatial study, but Q3 and Q5 full-size windows, the full-size recess case,
representative thickness/frequency coverage, and physical measurements remain
open. The local portable runtime is ignored and will not transfer to another
computer; that computer needs its own Python 3.12 environment and packages.

## Commands

Use fresh directories for every attempt. This computer's current Python 3.12.10
runtime is at `.venv/portable-runtime/python.exe`; other computers should use
their own project environment interpreter.

```powershell
.\.venv\Scripts\python.exe -m examples.embedded_wave --output outputs/embedded-smoke
.\.venv\Scripts\python.exe -m examples.verify_embedded_window --run outputs/embedded-smoke --output outputs/embedded-smoke-cpu
.\.venv\Scripts\python.exe -m examples.embedded_reference --order 4 --intact --output outputs/embedded-native-small
.\.venv\Scripts\python.exe -m examples.embedded_native_pulse --run outputs/embedded-smoke --reference outputs/embedded-native-small --output outputs/embedded-small-native-replay
.\.venv\Scripts\python.exe -m examples.embedded_wave --scope full --order 4 --steps 131072 --output outputs/embedded-full-intact
.\.venv\Scripts\python.exe -m examples.verify_embedded_window --run outputs/embedded-full-intact --output outputs/embedded-full-intact-cpu
.\.venv\Scripts\python.exe -m examples.embedded_wave_budget --order 4 --steps 294912 --output outputs/embedded-full-intact-fine
```

Add `--recess` for the material-loss state. Only the predeclared degree/time
pairs are accepted. Fixed-port numerical controls remain separate from the
seeded PZT mounting randomization in scenario-bank generation. Clamp
calculations are excluded; realistic placement/coupling variation remains
part of that scenario workflow.
