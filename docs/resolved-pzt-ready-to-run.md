# Ready to run the exploratory five-frequency comparison

The user requested completing preparation so they can launch independently.
The full five-experiment queue has **not** started. Only bounded numerical
checks and a 128-step full-size execution check were run during preparation.

## Start it yourself on this computer

Double-click `examples/run_resolved_pzt.cmd`. It checks inputs, opens the text
progress window, and launches 50 -> 100 -> 150 -> 200 -> 250 kHz serially.
Keep the scheduler window open. Closing only the progress viewer is safe;
the worker and queue logs continue recording. A duplicate launch is refused.

Alternatively, from the TRAXIS project folder in PowerShell:

```powershell
.\.venv\portable-runtime\python.exe -m examples.resolved_pzt_queue --run
```

For the read-only progress display in a separate terminal:

```powershell
.\.venv\portable-runtime\python.exe -m examples.pzt_terminal --root outputs/wandowski-resolved-pzt-v5 --resources --debug --transcript outputs/wandowski-resolved-pzt-v5/terminal-progress.txt
```

Output is `outputs/wandowski-resolved-pzt-v5/`. Its prepared receipt matches
the final source, configuration, data and readiness-evidence hashes. Preparing
again with identical inputs is harmless. Changed inputs, existing ownership
or previous experiment folders are rejected rather than overwritten.

The queue saves each waveform and final state before analysis, then saves and
prints comparison results before starting the next worker. A numerical or
analysis failure stops the queue. `all-results.json` is written at the end.
See `resolved-pzt-queue-reporting-20261117.md` for metric definitions and files.
No 300 kHz case is included: this specimen lacks that measured recording.

## Mesh readiness evidence

The old 24-sector/two-ring/two-layer body mesh was replaced by
**32 sectors, six radial rings, three thickness layers: 772 nodes per body**.
Further independent refinements used 32/7/3, 40/6/3, and 32/6/4.
The main free-disc resonance changed by 0.811%, 0.254%, and 0.401%, respectively.
Attached electrical driving-point response changed by at most 1.286%, 0.885%,
and 0.359% over the sampled bands surrounding the five center frequencies.
The engineering readiness targets are <1% resonance change and <2% attached
response change. These are exploratory criteria, not absolute error bounds.

The attached comparison uses a **fixed coarse 40 x 40 x 1 mm host patch**,
2 x 2 x 1 Q2 cells, and a Laplace evaluation with alpha=1/350 us to avoid
lossless poles. It does not add physical damping to the planned experiment.
It does not establish full-plate or receiver-waveform convergence, or agreement
with a physical PZT. The raw small-patch frequency responses and free-disc
studies remain in `outputs/pzt-readiness/`; compact metrics are in
`docs/evidence/pzt-body-readiness-refinement-final.json`.

The selected body mesh passed a real CUDA numerical check against independently
assembled CPU matrices: force difference 5.47e-16, mass inverse 2.97e-16,
eigenmode transient difference 7.00e-9, modified-energy drift 8.95e-15, and
CPU/GPU energy difference 3.04e-14. These are numerical consistency measures.

The full approximately 28-million-DOF plate with three selected bodies completed
128 steps in 28.99 seconds. Peak machine RAM was 73.8%, process RSS 2831 MiB,
GPU device memory 4693 MiB, and temperature 57 C. The required full-interval
step count remains 65536, with dt=350us/65536 below the checked stability
margin. The short check covers only 0.684 us; it is not a completed physical
experiment or a long-duration thermal test. Runtime guards remain enabled.

Refinement required increasing bounded body capacity from 512 to 1024 nodes,
and explicitly declared temporary attachment/merge workspaces to 2048 MiB.
The original 256 MiB component-assembly budget remains. A first full-size setup
attempt stopped at the former 1024 MiB merge forecast, before integration;
that failed attempt and receipt are preserved in the v4-check folder. The
successful retry is in v4-check2. These allowance changes did not lump mass,
reduce numerical precision, or change constitutive physics.

## Scope of readiness

`launch_ready=true` now means the exploratory batch can be executed. It does
**not** mean validated training data: `ready_for_training_data=false` remains.
The source experiment still has unidentified aluminum alloy, nominal elastic
properties, unmeasured bond/instrument details, and assumed free supports.
Perfect bonds, full-face electrodes, zero damping and ideal open-circuit
receiver inputs remain explicit model assumptions. New reference searches
did not replace the measured dataset. The larger research goal remains paused.

The proof `docs/evidence/resolved-pzt-run-readiness.json` binds selected mesh,
core implementation and passing refinement/CUDA/full-size evidence. Preflight
refuses modified evidence or source hashes. Do not bypass it to reuse old
output folders. Earlier runs/receipts are preserved.

Code and compact receipts can be backed up with Git. Raw reference recordings,
the local Python runtime and large outputs are ignored and are not transferred
by a GitHub clone; this launcher is prepared for the present computer.
