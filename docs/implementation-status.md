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
| 3D elastic propagation and backside loss | Implemented local linear-tet FEM; algebraic/dynamic checks and independent guided-mode dispersion convergence pass. Physical validation remains open |
| PZT electromechanics, supports, noise/coupling variation | Reference-solver pilot and measured calibration needed |
| Intact arrival/dispersion and mesh/time convergence | Rayleigh-Lamb reference and 3D FEM Bloch dispersion convergence implemented; finite-plate waveform/arrival convergence, resolved recess geometry and physical traces still needed |
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

## Independent guided-mode reference

`python -m traxis dispersion` reports phase/group velocities and wavelengths
for the nominal plate and its 1 mm remaining-thickness case. The implemented
Rayleigh-Lamb determinant removes tangent poles and trivial p/q roots. Tests
check the independent low-frequency extensional and flexural asymptotes,
traction residuals, frequency-thickness invariance, finite-difference group
velocity stability and the supported-band endpoints. Current suite: 33 tests.

The root search is limited to `1e-4 <= f * thickness / cT <= 0.25`, with only
fundamental A0 and S0 modes. It is an infinite, lossless isotropic-plate
reference; it must not substitute for finite-defect simulation. The
equations are grounded in [Perelli, section 1.3](https://amsdottorato.unibo.it/6321/1/Perelli_Alessandro_tesi.pdf).

A Bloch-periodic unit-cell check now compares the 3D tetrahedral FEM against
the independent A0/S0 traction roots at 100 kHz in 1 mm and 3 mm stock.
For each mode/thickness, 16/32/64/128 elements per wavelength are paired with
8/16/32/64 thickness cells. Frequencies converge with more than a factor
three error reduction per level; the finest A0 errors are about 0.304% (1 mm)
and 0.106% (3 mm). The verification report contains 19 numeric gates after
these additions. Separate x and z refinement tests also reduce error.

The first coarse A0 check failed the requested error tolerance: 32 elements
per wavelength still gave 4.72% error in 1 mm stock. Refinement was increased
rather than relaxing the accuracy requirement. This bending-resolution cost
is a concrete reason to prefer high-order/reference software for full-plate
datasets. The periodic benchmark is not the finite plate: it verifies mode
dispersion without edges, a recess or PZT coupling. The finite smoke waveform
retains all free-edge reflections and remains too coarse for training data.

