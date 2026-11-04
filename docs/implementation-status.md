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
| 3D elastic propagation and backside loss | Local FEM in development; stiffness, mass, rigid modes, patch test, dynamic energy and convergence checks needed |
| PZT electromechanics, supports, noise/coupling variation | Reference-solver pilot and measured calibration needed |
| Intact arrival/dispersion and mesh/time convergence | Independent numerical checks and later intact physical traces needed |
| Randomized specimens and exact depth labels | Geometry/split contracts in development; virtual dataset generator still needed |
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
| Resource bounds and telemetry | CPU-only short verification planned; system RAM, RSS, GPU temperature/VRAM where available; CPU temperature availability must be reported |
| Frequent commits on a project branch | Start with planning, then commit tested implementation increments |

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

