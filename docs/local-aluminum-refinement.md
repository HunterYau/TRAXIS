# Local aluminum refinement around PZTs

The separate adapter enables the requested 16-subdivision equivalent resolution:
12 mm target regions around each distinct PZT location, maximum target element
width 0.75 mm. The original host element order (fifth order in the current full
plate), thickness mesh, materials, PZT meshes, electrical equations and LTS
integration remain unchanged. This is not a change to the PZT's own rings.

The structured production grid previously supported only uniform elements.
`traxis/local_aluminum.py` removes entire coarse parent cells around each PZT
and replaces them with a refined tensor block. Everywhere outside the transition
blocks retains the exact original cells and coordinates. Top and bottom PZTs at
the same position share a single refined aluminum region.

Coarse-face breakpoints are retained and intervals are redistributed to avoid
short sliver cells. Consequently the current production alignment uses 18 inner
subdivisions at approximately 0.667 mm, rather than exactly 16 uniform cells.
This is slightly finer than the confirmed 0.75 mm target. Including the surrounding
transition, blocks have 20x20 and 18x20 elements. The transition bounds are
recorded in the prepared plan; the rest of the plate remains at its original
4 mm element width. These fifth-order cells differ from the second-order cells
in the earlier small 16x16 convergence experiment, so that experiment does not
constitute validation of the complete production model.

Hanging nodes use coarse-face interpolation. Fine faces never cross a coarse
trace breakpoint, so continuity holds between nodes, rather than only at the
interpolation points. Stiffness and mass both use the same transpose projection.
Projected boundary mass is retained as sparse non-diagonal blocks and solved
exactly; it is not lumped back onto a diagonal. No aluminum volume or mass is
double-counted. This follows the standard constrained finite-element approach
described by [deal.II's hanging-node documentation](https://dealii.org/developer/doxygen/deal.II/group__constraints.html).

`traxis/gpu_local_aluminum.py` reuses the original aluminum tensor arithmetic,
matrix-free PZT actions and cached exact FP64 factor solves. Fine/coarse local
actions are exposed through a separate adapter. The original optimized executor
code objects execute unchanged through isolated factory injection; no solver
module is patched or overwritten. Memory reuse, fused vector updates, CUDA
Graphs and the native cuSPARSE bridge remain available. The former uniform
implementation is retained unchanged as a reference and fallback.

`examples/local_aluminum_study.py` prepares and runs this opt-in configuration.
Old prepared studies/results are not silently reinterpreted with another mesh.

```text
python -m examples.local_aluminum_study --prepare --source-plan outputs/simulation-runs/spectral-q2-convergence-001/plan.json --root outputs/simulation-runs/spectral-q2-local-aluminum-001
python -m examples.local_aluminum_study --run --root outputs/simulation-runs/spectral-q2-local-aluminum-001
```

Only preparation was performed. Running is a separate explicit action. Each
completed kernel is checkpointed with a checksum before reconstructing the six
frequency records using the existing broadband reconstruction function. Failures
preserve completed kernels; no account or Pod lifecycle actions exist here.

Small verification includes fifth-order trace checks, continuity at points
between interface nodes, rigid
motion, stiffness symmetry, total mass, exact inverse mass, CPU/GPU agreement,
native solve graph capture, and 16 toy LTS steps against the unchanged CPU
recurrence. The tests are component checks, not full-plate simulations.

The prepared model is estimated to need about 21.3 GiB peak GPU memory with a
conservative setup/execution allowance. This exceeds the local 8 GiB GPU; use
suitable cloud hardware for a full run. The estimate and runtime admission checks
are not a measured peak or a cloud allocation check. Full-run speed and waveform
convergence are unmeasured. The local study uses exactly 4.9 ns fine and 9.8 ns
coarse steps for six rings, and 4.7 ns fine and 9.4 ns coarse steps for eight rings,
configured in `configs/local-aluminum-refinement.json`. Preparing the study stores
the selected pair in each case; execution reads that pair for each simulation.
Progress and kernel receipts record the selected pair. Coupled LTS stability and temporal
accuracy have not been certified for this spatial mesh. The prior explicitly
unchecked execution policy is preserved, not silently converted into a claim of
stability or temporal convergence.

The console records separate wall-clock stopwatches for the entire study, each
case, and simulation execution (including graph preparation inside the executor,
excluding host model assembly and waveform reconstruction). It prints and appends
every status update to `console.log`, checkpoints cumulative times in `timing.json`,
and writes `timing-summary.json` and `TIMINGS.txt` on case completion or failure.
Case results and kernel receipts also retain timing fields. On resume, accumulated
times include previous attempts and exclude time while the scheduler was stopped.
A forced process kill can lose timing since the last progress checkpoint.
The browser console shows both stopwatches and defaults to this local-refinement
study. Set `TRAXIS_STUDY_ROOT` to inspect a different remote study directory.

Each propagation update also prints and records `eta_remaining`, remaining
seconds, coarse steps per second, and the predicted UTC finish time. The ETA is
for propagation in the current simulation, excluding later reconstruction or
the next case. It uses up to 64 recent progress samples, excludes the initial
capture interval, and recalculates as speed changes. It says `Calculating...`
until three positive progress points are available and resets on each new case
or simulation attempt. The browser displays the same remaining-time estimate.
