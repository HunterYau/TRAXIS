# Implementation and evidence checklist

The source of requirements is [the consolidated research plan](https://docs.google.com/document/d/1YdcENA3Y00U-8t2Yn2oDq9uq08JuIf8LWrp96weIiNI/edit),
read 2026-11-03. A local snapshot is in `docs/research-plan-source.txt`.

Status means implemented and verified only when evidence is named here.
Full training and physical experiments are explicitly outside the short
debug runs authorized for this session; provide working code and honest
validation rather than inventing those outcomes.

| Requirement | Current state / acceptance evidence needed |
| --- | --- |
| Solver recommendation | Documented in solver-assessment.md; licensed solver pilot not run |
| 3D elastic propagation and backside loss | Implemented local linear-tet FEM; 25 tests and 11 numerical pilot gates pass. Physical/dispersion validation remains open |
| PZT electromechanics, supports, noise/coupling variation | Reference-solver pilot and measured calibration needed |
| Intact arrival/dispersion and mesh/time convergence | Independent numerical checks and later intact physical traces needed |
| Randomized specimens and exact depth labels | Analytic circular geometry, area-averaged depth labels, 8/12 sensor layouts and stable parent-specimen splits implemented/tested; dataset generator pending |
| Waveform bank indexed by specimen, TX, RX, frequency/settings | Pending; strict metadata, integrity checks and external solver import needed |
| Specimen-level splits and hidden observations | Pending; no ground truth or unrevealed traces in selector state |
| A: waveform CNN, attention, spatial decoder | Pending; partial masks/context, backward and serialization checks needed |
| B: DAS plus residual U-Net | Pending; baseline-subtracted imaging and raw observation context needed |
| C: DAS-only | Pending; calibrated depth conversion needed before reporting depth MAE |
| Ensemble uncertainty and calibration | Pending; independent members and error/coverage checks needed |
| Random, diagonal/edge fixed, RL adaptive | Pending; RL remains distinct from uncertainty heuristic |
| Equal-budget replay and fair architecture comparisons | Pending; identical cases/splits/subsets, repeated random seeds, dense reference |
| 128/256 outputs, extra decoder stage | Pending; isolated ablation matrix and common 512 reference |
| Quantitative and regional metrics | Pending; intact cases and false-positive/volume/depth/localization handling |
| Curves, 2D and 2.5D plots, parameter counts/cost | Pending |
| Simulation-only versus physical adaptation | Pending; import and adaptation workflow needed; two adaptation / three final damaged plates |
| Resource bounds and telemetry | CPU-only short pilot completed; cooperative time/RSS/system RAM/GPU guards tested. CPU temperature unavailable on this Windows host |
| Frequent commits on a project branch | research/elastic-tomography; planning commit followed by tested implementation increments |

## Provisional configuration

Keep both 8-PZT corner/edge and 12-PZT ring layouts configurable. Defaults
are computational choices, not settled hardware specifications. For five
frequencies, reciprocal acquisition contains 140 records for eight sensors
or 330 for twelve. Directed acquisition contains 280 or 660 respectively.
Reciprocity is an assumption to verify experimentally, especially after
electronics/coupling differences. Budget lists must derive from the action
catalog; never request 330 unique records from a 140-record bank.

Nominal elastic constants: E=68.9 GPa, nu=0.33, rho=2700 kg/m^3. Verify with
coupon/material documentation and intact-plate data. Frequencies, PZT size,
supports and acquisition sampling are still unresolved in the research plan.

## First numerical pilot (2026-11-03)

`python -m unittest discover -s tests -v`: 25/25 passed. `python -m traxis verify`:
11/11 numerical gates passed. Full report and traces are regenerated in
`outputs/verification/`. A compact evidence record is committed under docs/evidence.
No training was run. Dependencies passed `pip check`.

The 40 x 40 x 3 mm smoke plate used 405 intact nodes / 1,536 tets and 547 time
steps. The continuum-cube spatial error fell by factors 3.76 and 3.94; isolated
modal time error by 4.01 and 4.00. These are convergence checks for those
specific problems, not evidence that the smoke Lamb waves or defect are resolved.

The intentionally coarse circular recess has about 49% volume error from
cell-centroid removal. It is suitable for regression sensitivity checks only;
it must not generate research training data. The report explicitly sets
`ready_for_training_data=false`. Reference-solver geometry/mesh/time convergence
and intact-plate validation remain required.

Sampled pilot RSS stayed below 58 MiB; system RAM was about 85%; the GPU query
reported 1,724/8,192 MiB used and 47 C. The pilot allocated no GPU arrays. These
are sampled readings, not a continuous hardware peak measurement. CPU
temperature was unavailable. A hypothetical 250 x 250 x 6 cell local mesh
was screened at roughly 12.2 GiB assembly memory; no full-size mesh was allocated.

