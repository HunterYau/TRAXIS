# Adaptive sensing implementation

The project now has a learned RL selector and a separate uncertainty-path
heuristic. They operate on the same public partial-observation interface as
the random and fixed policies. No selector has been trained to convergence
or evaluated on validated elastic/physical data.

## Learned selector

`MaskedActorCritic` has a shared candidate-scoring actor and a state-value
baseline. Its state combines:

- A separate shared 1D CNN encoding only acquired raw/residual waveforms and
  their public geometry, frequency and sampling context.
- A small CNN encoding current predicted depth, ensemble spread and a flag
  indicating whether spread is available. Maps use a configured small grid.
- Each candidate's public TX/RX/frequency context, plus depth/spread sampled
  along its straight sensor path, and the acquired fraction of the catalog.

The policy has no access to specimen IDs, true depth/recess parameters,
unrevealed waveforms, rewards or reconstruction errors. Already acquired
actions have logit minus infinity. Empty states use a learned null record.
The candidate scorer shares weights, allowing variable action catalogs.
Tests cover an eight-PZT catalog and a twelve-PZT/two-frequency catalog;
functional support does not establish transfer accuracy between layouts.

The default requires ensemble spread. Setting `require_uncertainty=false`
is an explicit no-uncertainty ablation, with an availability flag rather than
pretending a missing spread map is calibrated zero uncertainty.

## Reward and optimizer

`PolicyTrainingSession` freezes the reconstruction ensemble. It trains only
on simulation training parents and uses validation parents for selector
selection. Final test/physical-holdout exposure is checked for both the
selector and inherited reconstruction lineage.

For each chosen record, reward is

`r_t = (E_before - E_after) / nominal_thickness_mm - record_cost`,

where `E` gives equal weight to mean depth error in true damaged and intact
regions. An intact control uses whole-plate MAE. Only the training/validation
orchestrator sees labels to calculate this quantity; the policy inputs do not.

The implementation uses finite-inspection reward-to-go policy gradients with
a learned value baseline and an entropy term, following the estimator in
[Spinning Up's policy-gradient derivation](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html).
Returns are `G_t = r_t + gamma*r_(t+1) + ...`; the actor uses
`-gamma^t * log(pi(a_t|s_t)) * stop_gradient(G_t - V(s_t))`. The critic regresses
to `G_t`; its gradient does not enter the actor advantage. The code is not
PPO and does not implement a clipped PPO objective or bootstrapped A2C.

Default gamma is 0.95, weighting improvements earlier in the inspection.
Episodes end at a configured acquisition budget. There is no learned STOP
action. At fixed horizon, a constant per-record cost cannot distinguish
which policy uses fewer records; measurement efficiency must be compared
using the equal-budget error curves. No autonomous stopping or hardware-time
savings claim follows from this reward. These settings remain provisional.

Collection uses no autograd, and then replays exactly the chosen public states
to accumulate gradients one state at a time. Network weights remain unchanged
until that replay finishes. This avoids storing a full episode's waveform
copies and convolution graphs while preserving the on-policy estimator.
Only compact spatial features and action/reward records are retained.

Debug mode permits at most four episodes, each with at most eight acquisitions.
Research mode rejects plumbing-only/arbitrary-unit data and debug reconstruction
artifacts. Policy learning currently runs on CPU. Optimizer, action/specimen
RNG state, best validation weights and pending validation are checkpointed;
source, dataset and reconstruction-artifact hashes are checked on recovery.

## Uncertainty-path heuristic

`UncertaintyPathPolicy` picks the available sensor path with highest sampled
mean ensemble spread. Same-path frequencies tie in catalog order; the heuristic
does not infer frequency informativeness. This straight-path approximation
does not model dispersive scattering or establish expected information gain.
It is a clearly named additional baseline, separate from the learned policy.

## Frozen replay

Evaluation plans accept `fixed`, seeded `random`, `uncertainty_path` and
checkpoint-bound `rl` policy specifications. An RL artifact is bound to the
exact reconstruction model/ensemble used to learn it. Changing that model,
the selector file, initial observation protocol or held-out lineage fails
validation. Compare selectors on one fixed reconstructor; prepare a separate
policy/plan for a different architecture. Uncertainty-path replay requires
an ensemble. Greedy RL evaluation does not receive reward or ground truth.

Current regression coverage includes finite actor updates, acquisition masks,
hidden-NaN isolation, action permutation equivariance, independent reconstruction
weights, analytic policy-gradient derivatives, nontrivial defect-region reward,
exact optimizer/action/weight recovery, validation interruption, matched replay
and provenance rejection. These checks verify implementation behavior and
do not establish that RL beats random/fixed sensing.

## Commands and software pilot

After creating the software fixtures/ensemble with `verify-workflow`, a small
policy run can use the example configuration:

```powershell
.\.venv\Scripts\python.exe -m traxis train-policy --dataset outputs/workflow-verification/dataset.json --reconstructors outputs/workflow-verification/direct-101.pt outputs/workflow-verification/direct-202.pt --config configs/debug-policy.json --episodes 1 --output outputs/policy-debug
.\.venv\Scripts\python.exe -m traxis train-policy --dataset outputs/workflow-verification/dataset.json --resume outputs/policy-debug/latest-policy.pt --episodes 1 --output outputs/policy-debug
```

This configuration permits two total episodes of three chosen records each.
`--episodes` is the current chunk, not a new total budget after each resume.
The default uses one CPU thread. No reconstruction weights are updated.

For selector comparisons, `freeze-evaluation` accepts `--rl-policies <policy.pt>`
and `--uncertainty-path` in addition to fixed/repeated random policies. Supply
the same two reconstruction artifacts and `--ensemble`. Both reconstruction
and selector hashes/lineage, initial observations and inference-source hashes
are checked before replay. A plan frozen under older inference code must be
run at that revision or replaced by an explicitly new plan.

The end-to-end adaptive pilot is:

```powershell
.\.venv\Scripts\python.exe -m traxis verify-policy --dataset outputs/workflow-verification/dataset.json --reconstructors outputs/workflow-verification/direct-101.pt outputs/workflow-verification/direct-202.pt --output outputs/policy-verification
```

Use a new/empty output directory. The pilot accepts only small, arbitrary-unit,
plumbing-only simulation fixtures, not research/physical holdouts. It performs
two RL debug updates with checkpoint recovery, then compares fixed, three
random seeds, uncertainty-path and greedy RL on both fixture test cases.
Budgets are 1/2/3 records plus the 28-record dense reference. All policies must
avoid repeats and converge to the same full-scan prediction. The dense budget
lies beyond this debug training horizon; this is a code-path check, not a
long-horizon generalization result. Confidence calibration remains open.

The final monitored pilot completed in 12.45 seconds. Its 17,530-parameter
selector received two debug updates; all 12 replay episodes passed mask,
budget and common-dense-prediction checks. Sampled process RSS stayed at most
388.6 MiB and system RAM at most 77.8%. GPU samples reported 1,664/8,192 MiB
used and 47 C; all computation was on CPU. CPU temperature was unavailable.
Compact evidence is in `docs/evidence/adaptive-debug.json`, with full reports
and maps under `outputs/policy-verification-final`. The current suite has
90 passing tests; there is still no research-quality RL performance result.
