# Reserved-parent uncertainty calibration

Ensemble standard deviation measures disagreement, not calibrated probability.
The optional post-hoc workflow fits prediction intervals using parents reserved
for `uncertainty_calibration`. Those parents cannot contribute to model/policy
fitting or model selection. Final simulation/physical test parents remain
separate. The physical `calibration` role still denotes the intact control;
it is not automatically reused for this statistical calibration.

The implementation follows the split-conformal finite-sample rank and scaled
residual construction in [Angelopoulos and Bates, sections 1.1 and 2.3](https://arxiv.org/abs/2107.07511).
Our structured-output score takes the maximum over one plate's reference
cells, recorded budgets and frozen policies. The coverage interpretation
requires independent exchangeable calibration/new parents, a fixed prediction
rule and matching protocol. These assumptions are not established by software
tests, metadata, numerical verification or an artificial fixture pilot.

## Scope and score

Let `mu(x)` and `sd(x)` be the frozen ensemble mean and sample standard
deviation in millimeters. At a predeclared positive floor `s0` in mm, one
calibration parent contributes:

```
score(parent) = max over cells, saved budgets and frozen policies:
                abs(true_depth_mm - mu_mm) / max(sd_mm, s0_mm)
```

The first implementation requires **one scan per parent** and the same
acquisition catalog, sensor coordinates, sampling, observable units, domain
and reference grid for every parent. Multiple policies are evaluations of
the same parent, not additional independent samples. Repeated scans and
incomplete/duplicated policy episodes are rejected. This restriction avoids
an unreviewed rule for unequal numbers of repeated scans.

For `n` parent scores and error rate `alpha`, select the
`ceil((n+1)*(1-alpha))`-th ordered score `q`. The code selects that order
statistic directly without interpolated quantiles. A decimal calculation
avoids rounding an integer rank upward accidentally. If the rank exceeds
`n`, store `q=null` and return the full physical interval `[0, nominal_mm]`.
Nine independent parents are the minimum for a finite 90% quantile; nineteen
are the minimum for 95%. Those minima do not demonstrate stable or useful
intervals; meaningful coverage assessment needs substantially more data.

For a new parent, bounds are:

```
lower_mm = max(0,          mu_mm - q * max(sd_mm, s0_mm))
upper_mm = min(nominal_mm, mu_mm + q * max(sd_mm, s0_mm))
```

A small float64 rounding allowance keeps exact score ties inside the bounds.
The target event covers every cell of the shared reference grid at every
listed saved budget and policy, for one new parent. It does not cover all
continuous points of the plate, unrecorded budgets, other action paths or
new policies. Prediction intervals are not confidence intervals for model
parameters, per-plate conditional probabilities or certified defect bounds.

Clipping to a known physical range does not fix incorrect geometry labels.
No Gaussian assumption is made. The score remains valid as a conformal
construction even with a poor uncertainty estimate, but its intervals can
be too wide to be useful. Such poor outcomes must be reported.

## Freeze before calibration replay

Reserve the role before simulation generation or physical import. The study
configuration reserves an expected 10% of all parents by carving them out of
the original 70% training allocation. Validation/final-test assignments stay
fixed. `SamplingSettings.uncertainty_calibration_fraction` defaults to zero for
compatibility; `configs/scenarios-study.json` explicitly sets it to 0.1.
Actual counts remain approximate and appear in the unsolved recipe plan.
Nuisance variants retain their parent's role, but select only one scan per
parent for this calibration workflow.

Select models, policies, thresholds, `alpha` and floor without looking at
calibration labels or final outcomes. Then:

```powershell
.\.venv\Scripts\python.exe -m traxis freeze-evaluation --dataset data/dataset.json --models checkpoints/member-1.pt checkpoints/member-2.pt --ensemble --split uncertainty_calibration --budgets 10 20 40 80 --random-seeds 17 29 43 --output outputs/calibration-replay-plan.json
.\.venv\Scripts\python.exe -m traxis freeze-calibration --evaluation-plan outputs/calibration-replay-plan.json --alpha 0.1 --std-floor-mm 0.05 --mode research --output outputs/calibration-plan.json
.\.venv\Scripts\python.exe -m traxis evaluate --dataset data/dataset.json --plan outputs/calibration-replay-plan.json --calibration-plan outputs/calibration-plan.json --output outputs/calibration-replay
.\.venv\Scripts\python.exe -m traxis fit-calibration --plan outputs/calibration-plan.json --evaluation outputs/calibration-replay --output checkpoints/depth-intervals.json
```

The 0.05 mm floor is a provisional computational setting, not a measured sensor
limit. Choose it before the calibration cohort is inspected. Reserved replay
requires the frozen calibration plan and binds its fingerprint into the saved
result. Fitting rejects a different floor/alpha/mode plan against that result.
The code enforces this sequence; it cannot establish whether a person already
inspected labels or intentionally reran selection on the same reserved cohort.
Research mode rejects debug checkpoints, arbitrary-unit or plumbing-only data.

The scalar calibration artifact binds the model/selector artifact hashes,
inference source, complete finite protocol, acquisition/grid/domain signature,
calibration source, parent identities, scores and order statistic. Damaged
geometry labels are used only in this post-hoc evaluator. Raw ensemble spread
still enters the frozen selector; replacing it with interval widths would
change the acquisition protocol and invalidate this calibration.

## Independent assessment

Freeze and replay the **same** inspection protocol on simulation `test` or
`physical_holdout` parents, then assess it:

```powershell
.\.venv\Scripts\python.exe -m traxis freeze-evaluation --dataset data/dataset.json --models checkpoints/member-1.pt checkpoints/member-2.pt --ensemble --split test --budgets 10 20 40 80 --random-seeds 17 29 43 --output outputs/test-replay-plan.json
.\.venv\Scripts\python.exe -m traxis evaluate --dataset data/dataset.json --plan outputs/test-replay-plan.json --output outputs/test-replay
.\.venv\Scripts\python.exe -m traxis assess-calibration --calibration checkpoints/depth-intervals.json --evaluation outputs/test-replay --output outputs/test-intervals
```

`coverage.json` reports each parent's joint coverage, per-map pixel and defect
coverage, interval width and full-physical-span fraction. Lower/upper arrays
are saved in checksummed per-case NPZ files in mm. The `passed` field means
the software completed; `empirical_coverage_meets_nominal` records whether
the observed fraction reached the target. Neither is a statistical test of
population coverage. Report actual parent counts and both coverage and width;
do not tune settings or reject difficult cases after final outcomes are seen.

Assessment refuses reused calibration parents and changes to models, policies,
random seeds, initial actions, budgets, thresholds, inference source, domain,
sampling or grid. A simulation calibration cannot be applied to physical
voltage traces. The present six-plate plan supplies no extra independent
damaged calibration cohort; do not borrow the three final physical plates.
Additional physical calibration requires a separately designated cohort and
appropriate matching acquisition/distribution evidence. Even under an otherwise
valid method, two or three plates cannot support a finite 90%/95% quantile.

## Short software pilot

After `verify-workflow`, reuse its existing debug artifacts:

```powershell
.\.venv\Scripts\python.exe -m traxis verify-calibration --dataset outputs/workflow-verification/dataset.json --reconstructors outputs/workflow-verification/direct-101.pt outputs/workflow-verification/direct-202.pt --output outputs/calibration-verification
```

This creates nine clearly labeled calibration fixture parents, replays fixed
and seeded-random policies, fits one scalar rank and assesses the two original
test fixtures. It runs **zero training updates**. Source recipes, checksums,
resources and full diagnostic outcomes are retained. Artificial fixture IDs
are not evidence of independent physical specimens or exchangeability.

The first pilot's average interval width was 2.99 mm in 3 mm stock and only
one of two held-out fixture parents was jointly covered, despite a nominal
90% target. Its settings were not retuned after that outcome. This is a
software success with unusable uncertainty quality, not a calibrated research
result. Useful coverage and precision remain unverified until models and
appropriate independent data are available.
