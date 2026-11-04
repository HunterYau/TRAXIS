# Matched architecture studies

The study runner plans a factorial experiment, advances it in explicit bounded
chunks, checks actual observation exposure and freezes comparisons. Planning
and freezing run no optimizer updates. The supplied debug configuration is a
software check, not a research training schedule.

## Plan without optimization

```powershell
.\.venv\Scripts\python.exe -m traxis plan-study --dataset data/dataset.json --config configs/study-debug.json --output outputs/study
```

The example combines Model A/B, 128/256 outputs, zero/one extra decoder block
and two paired trials. Weight seeds 101/202 pair with subset seeds 37/53. All
eight configurations in trial 101 see the same training draws; trial 202 uses
its own matched draws. It plans 16 neural jobs, four DAS fits (one per output
size/trial), and 24 pairs differing in exactly one architectural factor.

Widths, attention settings, optimizer/loss settings, nominal depth, eligible
parents, normalization and validation subsets remain common. The planner reads
only train/validation banks, requires a shared 512 x 512 reference grid and
nominal thickness, and fits one train-only signal RMS for all neural members.
Reserved calibration and final-test labels are not read. Replicates are not
counted as additional independent specimens.

Review `study.json` before advancing it. It records resolved configurations,
data/source fingerprints, signal scale and controlled pairs. Keep its directory,
dataset location and recorded source revision for exact recovery. Missing or
incompatible budgets, units, group speeds or validation parents are rejected.
Research mode rejects plumbing-only/arbitrary-unit data; metadata alone does
not prove physics validity. The example's 2,500 m/s and 25 us imaging settings
belong only to the analytic fixture and need control/training calibration for
real measurements.

Both intact and damaged training parents are required. An intact validation
or reserved calibration plate cannot replace a training control.

## Advance a bounded chunk

```powershell
.\.venv\Scripts\python.exe -m traxis advance-study --plan outputs/study/study.json --max-new-updates 1
.\.venv\Scripts\python.exe -m traxis advance-study --plan outputs/study/study.json --jobs A-r128-extra0-seed101 --max-new-updates 1
.\.venv\Scripts\python.exe -m traxis advance-study --plan outputs/study/study.json --max-new-updates 0 --baseline-fits 1
```

The default applies at most one new neural update across selected jobs and
fits no baseline. Jobs run in plan order; select IDs to interleave them. A
debug study allows at most 32 total planned updates, each session keeps its
eight-update debug cap, and research jobs have at most 100,000 explicit updates.
There are at most 64 neural jobs. No epoch loop or GPU choice is inferred.

One job runs at a time. An OS file lock prevents concurrent advances/freezes
of the same study; the persistent `.study.lock` file is not a live-process
claim. Windows uses the [nonblocking byte-range lock](https://docs.python.org/3/library/msvcrt.html#msvcrt.locking).
Separate studies still need separate resource limits if launched concurrently.

Each job owns its checkpoint, progress record and selected artifact. A run ID
binds the study fingerprint and job ID, rejecting foreign checkpoints even
with identical hyperparameters. Completed jobs are checked and reused without
another update. Unknown files in an unowned job directory are not overwritten.
Progress retains loss/validation histories, parameters, elapsed compute time,
and hash prefixes of exact specimen/action draws. Prefixes must agree among
variants in each paired trial; mismatches block artifact export/freezing.
Checkpoint history retains the actual action IDs. Compute time includes
initialization, preprocessing, validation and I/O, not just optimization.

For the primary ablation, early-stopping patience is raised above the common
update budget and the resolved value is recorded. Every job must finish that
budget before freezing. Each selects its best weights from the same validation
schedule; selected steps may differ and are recorded. The final update is
always validated. If that validation is interrupted, recover without another
optimizer update using `advance-study --max-new-updates 0 --jobs <job-id>`.
The individual trainer also supports `train --resume ... --steps 0` for this
case while retaining export overwrite protections.

DAS fits require `--baseline-fits`. They use all training parents at planned
budgets/subset seeds, with shared reference labels and per-image normalized
weights. Their draws match across DAS resolutions; they are not the neural
optimizer's sampled history. Report their exposure and cost separately from
gradient fitting. Scalar and neural procedures do not perform equal work.

## Freeze isolated comparisons

```powershell
.\.venv\Scripts\python.exe -m traxis freeze-study --plan outputs/study/study.json --split validation --budgets 2 4 --random-seeds 17 29 43 --output outputs/study-comparisons
```

All jobs must be complete, with matched exposure, intact artifact hashes and
consistent provenance. Freezing writes:

- `individuals.json` for batch evaluation of all scalar/neural artifacts.
- `controlled-pair-*.json` for architecture, resolution or decoder-depth
  pairs. Architecture pairs include the matching-resolution/trial DAS fit
  when one was planned.
- `ensemble-*.json` for independently initialized members of each identical
  architecture configuration, sharing normalization and imaging settings.
- `comparisons.json` mapping predictors to jobs/factors and retaining
  parameters, compute/loss/validation records and fingerprints.

Find pair filenames by factor/job IDs in `comparisons.json`; their numeric
suffixes are ordering details. All plans share initial actions, budgets,
thresholds and fixed/repeated-random policies. Paired plans keep plots readable;
the all-artifact plan may exceed the reporter's 24-series display limit.

```powershell
.\.venv\Scripts\python.exe -m traxis evaluate --dataset data/dataset.json --plan outputs/study-comparisons/controlled-pair-000.json --output outputs/pair-replay
.\.venv\Scripts\python.exe -m traxis report --evaluations outputs/pair-replay --labels controlled-pair --output outputs/pair-figures
```

Use validation results to select settings before final simulation testing.
A predeclared full-grid simulation ablation may use `--split test`; do not tune
it after seeing that outcome. Physical evaluation remains a separate stage for
selected configurations under the two-adaptation/three-holdout protocol.
The runner does not sweep configurations on those final physical plates.

Evaluate adaptive sensing after reconstruction selection with the existing
`train-policy`/`freeze-evaluation` commands. RL policies remain bound to their
chosen model/ensemble and initial acquisition protocol. Architecture sweeps use
fixed/random observations so selector changes cannot masquerade as architecture
gains. Reserved-parent interval calibration likewise follows model selection.

## Short verification

After `verify-workflow` has generated its labeled fixtures:

```powershell
.\.venv\Scripts\python.exe -m traxis verify-study --dataset outputs/workflow-verification/dataset.json --output outputs/study-verification
```

This performs one update per neural job, four DAS fits and freezes all 24
isolated pairs/eight ensembles. It replays the first predeclared pair for each
factor and one ensemble on validation fixtures, checking matched acquired
actions. Other plans remain unexecuted. Every neural member exercises forward,
backward, validation and serialization; the pilot does not establish trained
architecture quality. Cooperative time/RAM/system-memory/VRAM/temperature
guards apply between bounded operations. CPU execution, recovery and Windows
locking are tested; CUDA needs its own measured resource/reproducibility checks.
