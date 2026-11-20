# Selected next experiment: restored ideal properties

The default configuration is now `configs/pzt/resolved-wandowski-ideal-restored.json`.
It restores the earlier perfect-bond, ideal-open receiver, unit-peak drive and
zero-damping settings for all five frequencies. Its separate output folder is
`outputs/simulation-runs/run-010-wandowski-restored-ideal-properties/`.
See [the restored setup](restored-ideal-experiment.md). The queue has been prepared,
but no new simulation was started. The default launcher and progress viewer select run 010.

The following sections describe the retained, completed run 009 realistic setup.

## Historical realistic-assumption exploratory comparison

The current v9 setup adds finite bond stiffness/mass, a finite resistance and
capacitance receiver circuit with mechanical backreaction, and an assumed
X10 amplifier with estimated delay. These are documented plausible settings,
not measured settings of the original experiment. See
[assumptions and evidence](wandowski-realistic-assumptions.md).

The v9 configuration completed 50, 100, 150, 200 and 250 kHz with an
explicitly assumed 6061-T6 damping proxy. Its results now live in
`outputs/simulation-runs/run-009-wandowski-realistic-bond-damping/`.
Earlier undamped results are in `run-006-wandowski-ideal-bond-50-100khz`
and `run-007-wandowski-ideal-bond-150-250khz-continuation` under `outputs/simulation-runs/`.
See [output organization and restarting](plate-output-organization.md).

See [the damping implementation and assumptions](aluminum-damping.md).
The Wandowski alloy is unidentified. This is an effective narrowband damping
estimate, not measured damping of that plate or proof of improved accuracy.
The damping proxy applies to the whole coupled plate/PZT system; separate losses
in aluminum, PZT, adhesive and instruments remain unresolved.

Resource limits remain monitoring-only: no automatic RAM, GPU memory,
temperature, time or log-size aborts. Cooling pauses at 82 C and resumes
at 78 C. Missing readings are retried without terminating the calculation.
Numerical validity, stable time steps, provenance and saved-data checks remain.

## Start it yourself on this computer

Double-click `examples/run_resolved_pzt.cmd` to start/resume the selected run 010.
It checks saved results, opens the text progress window, and resumes pending frequencies
in the same experiment folder. Run 009 is already complete and remains separately saved.
Keep the scheduler window open. Closing only the progress viewer is safe;
the worker and queue logs continue recording. A simultaneous duplicate launch is refused.

Alternatively, from the TRAXIS project folder in PowerShell:

```powershell
.\.venv\portable-runtime\python.exe -m examples.resolved_pzt_queue --run
```

For the read-only progress display in a separate terminal:

```powershell
.\.venv\portable-runtime\python.exe -m examples.pzt_terminal --resources --debug
```

Historical run 009 configuration is `configs/pzt/resolved-wandowski-v5.json`.
Its output is `outputs/simulation-runs/run-009-wandowski-realistic-bond-damping/`.
Its original prepared receipt is preserved. Reporting/organization code has since
changed: a fresh simulation batch needs a new folder and a new `--prepare` receipt.
Resuming pending tests requires identical prepared inputs; completed tests are
verified and retained. Interrupted frequency attempts are archived in that frequency
folder before retrying from the beginning. See the organization guide for commands.

The queue saves each waveform and final state before analysis, then saves and
prints comparison results before starting the next worker. A numerical or
analysis failure stops the queue. `all-results.json` is written at the end.
See `resolved-pzt-queue-reporting-20261117.md` for metric definitions and files.
No 300 kHz case is included: this specimen lacks that measured recording.

## Numerical checks and limits

The current code passed finite-interface rigid-motion, mass and uniform-slip
checks; an independent coupled RC transient ODE comparison on CUDA;
second-order temporal refinement; and undamped/damped zero-port regressions.
The final full plate with three PZTs, the finite adhesive interfaces and
receiver loads completed 128 steps in 38.93 s. This covers only 0.684 us.
The planned full interval remains 350 us with 65536 steps.

The PZT bodies retain 32 sectors, 6 rings, 3 layers (772 nodes). Previous
perfect-bond body mesh studies are historical context. They do not establish
waveform convergence of the new finite bond or full plate. Training acceptance
and measured agreement remain unverified. The current thin interface has
finite stiffness and mass but does not resolve the adhesive volume, its
geometric offset, nonuniform coverage or adhesive loss.

The 4096 MiB attachment/merge allowance accommodates the additional retained
body degrees of freedom. It does not reduce precision or lump PZT mass.

## Scope of readiness

`launch_ready=true` now means the exploratory batch can be executed. It does
**not** mean validated training data: `ready_for_training_data=false` remains.
The source experiment still has unidentified aluminum alloy, nominal elastic
properties, unmeasured bond/instrument details, and assumed free supports.
Finite approximate bonds, full-face electrodes, assumed effective damping and
assumed finite receiver loads remain explicit model assumptions. New reference searches
did not replace the measured dataset. The larger research goal remains paused.

The proof `docs/evidence/resolved-pzt-organized-output-readiness.json` inherits
`docs/evidence/resolved-pzt-realistic-readiness.json` and records the verified
output-path-only changes to two PZT evidence scripts. It binds selected mesh,
core implementation and current numerical/CUDA/startup evidence. Preflight
refuses modified evidence or source hashes. Do not bypass it to reuse old
output folders. Earlier runs/receipts are preserved.

Code and compact receipts can be backed up with Git. Raw reference recordings,
the local Python runtime and large outputs are ignored and are not transferred
by a GitHub clone; this launcher is prepared for the present computer.

## Combined text results

The queue writes `all-results.txt` in its output folder at startup and refreshes
it after each worker exits, including failures. It contains the saved experiment
summaries, comparison results, uncertainty labels, elapsed times and completion
counts, without progress or hardware telemetry logs. Existing per-experiment
files and `all-results.json` are retained.

Export or refresh this file for a saved batch without launching simulations:

```powershell
.\.venv\portable-runtime\python.exe -m examples.pzt_terminal --export-results
```

As with other queue source changes, future simulation launches need a freshly
prepared output version; do not rewrite the old run's frozen source receipt.
