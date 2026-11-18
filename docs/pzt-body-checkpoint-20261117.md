# Resolved PZT body checkpoint — November 17, 2026

Work saved after the user reported 16% remaining of the five-hour allowance.
**No full experiment or training is running.** The text status window was
opened, and the bounded 128-step full-size check has finished successfully.
The five-frequency scheduler exists and is prepared, but its long-run gate
is closed. Do not represent this checkpoint as five new experimental results.

## Added and connected

`traxis/pzt_body.py` creates a bounded linear-tetrahedral disk using the existing
nominal NCE51 full material tensor. It retains the PZT body's 3D mass, stiffness,
piezoelectric coupling and condensed electrostatics. Its contact face follows
the plate's surface interpolation. Applying the same transformation to force,
stiffness and consistent mass makes the connection reciprocal. Neither a
point mass nor diagonal body-mass lumping replaces the ceramic mechanics.
Overlapping interpolation supports are merged before mass inversion; physically
overlapping bodies on the same surface are rejected.

`traxis/gpu_pzt_body.py` connects those bodies to the full tensor plate on CUDA.
The Verlet solver uses the coupled mass inverse and the corresponding kinetic
and modified energy expressions. A bounded float64 matrix-vector CUDA kernel
avoids requiring a missing cuBLAS-Lt DLL in this project's portable runtime;
no software was installed and no precision was reduced. CPU/GPU checks cover
the new kernel and coupling. All per-step states stay on the GPU.

Electrical cases are prescribed voltage and ideal open circuit (optionally
parallel capacitance). Open-circuit voltage feeds back mechanically through
the same coupling coefficients. Finite-resistance instrument dynamics are
not implemented in this coupled-body path and are rejected. The older
one-way RC observer still exists, but it is not silently substituted here.

The formulation uses the existing stress-charge equations, consistent with
[COMSOL's primary theory documentation](https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_theory.06.043.html).
This is our implementation, **not a COMSOL run or COMSOL code**. Nominal NCE51
tensors remain from the cited Noliac Ver1404 profile. The incomplete STEMINC
profile cannot silently become a full 3D material.

## Assumptions and remaining work

The attachment is an explicitly assumed perfect bond. No measured glue
thickness, stiffness, loss, wrap-around electrode, solder/wire mass, amplifier
transfer or cable load has been invented. The prototype can produce body
resonances, but their accuracy depends on mesh and material/attachment data.
It is not calibrated against experimental resonance or absolute voltage.

Free-disc refinement found the dominant electrically excited resonance at
212.93, 200.92, 195.72 and 195.46 kHz on progressively changed meshes.
The last two differ by about 0.134%, **but both have three radial rings**;
this checks angular refinement, not radial/thickness convergence. The
lossless response near 200 kHz is sensitive to those changes. The current
24-sector/two-ring/two-layer queued mesh therefore needs refinement before
long comparisons. A free disc is also not a bonded plate: perform a bounded
attached-body refinement check before interpreting improved error percentages.

Next: vary radial rings and thickness layers independently within the memory
budget; assess the driven/received response over the actual pulse bandwidth,
then freeze a new mesh/configuration/output version. If glue or instrument
properties remain missing, label assumptions and sensitivity ranges explicitly.
Do not select arbitrary ranges or tune them to make the reference error small.

## Tests and preserved evidence

- Full suite: 490 tests run, 35 skipped, no failures (455 passed).
- After final small changes: 28 targeted checks passed.
- Small real CUDA check: force difference 1.84e-16, mass inverse difference
  2.62e-16, eigenmode transient difference 5.75e-8, modified-energy drift
  8.65e-15; independent CPU energy difference 1.91e-14. These are numerical
  consistency checks, not real-world accuracy percentages.
- Full-size 28-million-DOF plate, three bodies, 128 steps: completed in 11.46 s.
  Peak system RAM 71.8%, process RSS 1942 MiB, GPU memory 4714 MiB, GPU 55 C.
  This checks startup/execution only: simulated duration 0.684 microseconds,
  **not** the intended full 350 microseconds. It does not validate arrivals.
- Raw short-run files and exact frozen source ZIP:
  `outputs/wandowski-resolved-pzt-v1/`. The final v2 prepared queue is separate.
- Compact receipts: `docs/evidence/pzt-body-{cpu-gpu-check,full-size-smoke,free-disc-refinement}-20261117.json`.
- Previous full-plate code: commit `42e619e`, plus local source archive
  `outputs/pzt-receiver-before-body-42e619e.zip`. Completed old waveforms intact.

The old prepared queue hashes all `traxis/*.py`. New core code intentionally
changes that identity; do not disable the hash check to reuse old output folders.
Use the saved original source/commit to reproduce old runs.

## Scheduler and display

`examples/resolved_pzt_queue.py` schedules 50/100/150/200/250 kHz against
Wandowski plate 1's measured PZT channels. The paper/dataset search found no
replacement demonstrably more complete for this task; see the search audit.
Using this same dataset would provide a controlled before/after comparison,
not a new plate validation. Its remaining reconstruction gaps stay explicit.
300 kHz is excluded because plate 1 has no matching recording.

Configuration: `configs/pzt/resolved-wandowski-v1.json`.
Prepared output: `outputs/wandowski-resolved-pzt-v2/prepared.json`.
`launch_ready=false` makes `--run` refuse before spawning or taking ownership.
The source hash will also refuse if anything changes after preparation.
Resolve the documented mesh/referral choice before preparing a fresh version;
never overwrite an existing frozen receipt.

Each future completed experiment saves waveforms/final state and automatically
computes the same paired fast/slow timing comparison. Lower percentages mean
smaller flight-time differences. Slow-wave candidates above the paper's reported
PZT detection range remain explicitly unresolved rather than becoming 0% errors.
Failure stops the queue; `STOP_AFTER_CURRENT` stops between completed runs.
No intermediate-state restart has been added.

`examples.pzt_terminal --resources --debug --root OUTPUT` displays progress,
steps, elapsed/ETA, current frequency, log location, machine RAM, GPU use,
device memory and temperature. Shared GPU memory is not measured. This is a
read-only terminal progress/debug-log display, not a breakpoint debugger.
The visible setup-check window can be closed; no simulation depends on it.

The older broad goal remains paused. No cloud jobs, emails or purchases.
