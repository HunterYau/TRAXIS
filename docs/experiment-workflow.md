# Dataset, training and evaluation workflow

These commands implement the experiment machinery. They do not establish a
validated forward model or a trained reconstruction result. Debug mode is
limited to eight total updates per session and accepts explicitly labeled
software fixtures. Research mode rejects plumbing-only/arbitrary-unit data.
A status field is a provenance declaration, not proof of independent physics
validation; the solver acceptance work in solver-assessment.md remains required.

## Index waveform banks

Place banks under the directory containing their dataset manifest:

```powershell
.\.venv\Scripts\python.exe -m traxis index-data --banks-dir data/banks --output data/dataset.json
```

Indexing checks bank integrity and parent-level split separation, records
file hashes, and reads one bank at a time. Later reads reject changed files
or metadata. The manifest contains scan IDs and parent IDs; replay/model
inputs do not. Repeated scans of one physical plate are one independent parent.

## Bounded subset training

```powershell
.\.venv\Scripts\python.exe -m traxis train --dataset data/dataset.json --config configs/debug-direct.json --output outputs/direct-debug --steps 2
.\.venv\Scripts\python.exe -m traxis train --dataset data/dataset.json --resume outputs/direct-debug/latest.pt --output outputs/direct-debug --steps 1
```

The supplied configs are debug examples. The DAS example's 2,500 m/s speed and
25 us offset belong to the analytic software fixture, not a calibrated plate.
Choose real group speeds, offsets and frequencies using control/training data.

Each update draws one training specimen and a random subset size/action set.
`seed` controls weight initialization; `subset_seed` independently controls
specimens and observations, allowing architectures to share identical data
draws. Compact model tensors contain only acquired records, reducing partial
inspection cost. The complete permitted action catalog still belongs to the
public replay state for policy selection. Empty subsets use the learned prior.

Adam state, model weights, CPU/CUDA RNG state, subset RNG, completed steps,
validation history and best weights are atomically checkpointed. Exact
resume requires the same dataset fingerprint and training source revision.
The checkpoint preserves a completed update even if the following validation
is interrupted; pending validation runs before another update after recovery.
It will not silently overwrite a different session's checkpoint.

Validation uses fixed subsets of validation parents only. Best weights and
early-stopping patience depend on those losses, never final test plates.
Exports record fit/validation parent IDs, selected step, scale, imaging
settings and source provenance. The maximum total step budget is explicit in
configuration; `--steps` controls only the current chunk. There is no implicit
100-epoch loop. Long research training is intended for the user after data and
configuration review, and has not been run in this session.

Default commands use cooperative 60-second / 1,024 MiB process-memory bounds,
system RAM limits and available temperature/VRAM telemetry. CUDA execution is
an explicit configuration choice and requires a suitable PyTorch installation.
Only CPU execution and recovery have been tested here. Use an appropriate
resource limit after measuring the actual model/data cost; a guard cannot
preempt a single blocked native kernel.

## DAS-only quantitative baseline

```powershell
.\.venv\Scripts\python.exe -m traxis calibrate-das --dataset data/dataset.json --settings configs/debug-das-settings.json --budgets 2 4 --output outputs/das-debug.json
```

Raw DAS intensity is not depth. This command fits a nonnegative through-origin
scalar conversion from signal units to mm using training specimens/subsets.
Both intact and damaged training cases are required. The weighting and budget
list are recorded. This empirical calibration does not account for complex
nonlinear depth/scattering effects; its quantitative error is a baseline to
measure, not a claim of a physical inversion. Validation/test labels cannot
fit the scale. Model C contains no neural network.

## Freeze and run equal-budget comparisons

```powershell
.\.venv\Scripts\python.exe -m traxis freeze-evaluation --dataset data/dataset.json --models outputs/direct-debug/selected-step-2.pt outputs/das-debug.json --split test --budgets 2 4 --random-seeds 17 29 43 --output outputs/frozen-debug-plan.json
.\.venv\Scripts\python.exe -m traxis evaluate --dataset data/dataset.json --plan outputs/frozen-debug-plan.json --output outputs/replay-debug
```

A plan binds model and dataset hashes, split, initial actions, thresholds,
policies and seeds before replay. Execution loads the bound artifacts and
checks their actual fit/model-selection lineage against held-out parents.
Changing a checkpoint or a plan invalidates its fingerprint. Use validation
results for model/policy/configuration selection, then freeze settings before
test/physical_holdout. Freezing is a reproducibility operation, not proof that
someone did not previously inspect the final outcomes.

The runner evaluates each neural/DAS artifact under fixed and repeated random
policies, with identical initial observations and record budgets. It also
evaluates the full available catalog as a dense prediction reference. Invalid
budgets fail: an eight-PZT reciprocal five-frequency bank has 140 records,
not 330. The full scan is never ground truth. Near-zero dense error produces
no percentage ratio; absolute excess error is always reported.

Each episode saves maps and metrics, timing, acquired action IDs and excitation
request counts. Seeds/scans average within parents before across-parent
summaries. Missing-defect centroid errors remain undefined, not zero. Virtual
replay reports no measured hardware acquisition time. The debug-only flag
remains true if any model or test data is a software fixture.

For ensembles, pass independently initialized/trained neural checkpoints with
the same architecture/scale/imaging settings and `--ensemble`. Distinct artifact
hashes and recorded seeds are required. Ensemble standard deviation and
error/coverage diagnostics are recorded; calibrated coverage is not assumed.

## Physical adaptation

Prepare an explicit physical dataset containing only designated roles:
`calibration`, `adaptation`, and `physical_holdout`. Use two damaged parent
plates for adaptation and reserve three different damaged parents for final
evaluation, as in the research plan. Repeated scans retain their parent's role.
Create a configuration with `training.phase="adaptation"` and matching model
parameters, then use `train --adapt-from <simulation-model.pt> ...`.

Adaptation preserves the simulation signal scale, requires matching observable
units, and selects updates only from adaptation parents. It records the source
checkpoint hash and inherited model-selection lineage. It refuses an initial
model already exposed to a final physical parent. This workflow has been tested
with synthetic role fixtures; no actual physical adaptation has been run.

## Remaining work

Adaptive RL and uncertainty-guided policies, uncertainty calibration,
automatic ablation/ensemble orchestration, research-quality scenario/response
generation, plotting, and intact physical validation remain open. Passing
these workflow tests does not select a scientifically superior model.

## End-to-end software pilot

```powershell
.\.venv\Scripts\python.exe -m traxis verify-workflow --output outputs/workflow-verification
```

Use a new/empty directory for each pilot. This command creates five clearly
labeled analytic fixture banks, indexes their parent roles, runs four total
debug updates across three small neural instances, exercises checkpoint resume,
and fits the DAS scalar conversion on training fixtures. It then runs 18
matched A/B/C fixed/random episodes plus four two-member ensemble episodes.
The test cases include an intact and a damaged fixture; neither is a real
plate or a validated elastic simulation. All methods receive the same acquired
records at matched budgets. Different debug update counts exercise recovery;
they do not constitute a fair comparison of trained architecture quality.

The first pilot completed in 16.56 seconds, with sampled process RSS at most
384.2 MiB, system RAM at most 76.5%, GPU usage 1,669/8,192 MiB and GPU temperature
47 C. The computation used CPU only; CPU temperature was unavailable. The
full report/maps remain under outputs; compact evidence is committed in
`docs/evidence/workflow-debug.json`. Both the aggregate and per-case results
are labeled debug-only. No full training cycle or physical adaptation ran.
