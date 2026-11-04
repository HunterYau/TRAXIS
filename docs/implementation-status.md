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
| Randomized specimens and exact depth labels | Seeded geometry/nuisance recipes, intact controls, nine size/depth combinations, clearances, parent-preserving variants and area-averaged labels implemented/tested; validated research dataset pending |
| CAD-conforming geometry and elastic bank generation | Gmsh/OpenCASCADE recess mesh with volume/refinement checks and bounded grouped 3D elastic solves/cache implemented; banks remain plumbing_only pending wave/PZT/physical validation |
| Waveform bank indexed by specimen, TX, RX, frequency/settings | NPZ schema, per-array integrity, bounded loading, explicit scalar CSV external-solver/DAQ import and source hashes implemented/tested; full physical/FEA dataset pending |
| Specimen-level splits and hidden observations | Fingerprinted lazy dataset, role-aware subset training/adaptation and checkpoint-bound held-out evaluation enforce parent separation; masked replay tested |
| A: waveform CNN, attention, spatial decoder | Implemented with geometry/sampling/amplitude context, partial masks and empty-state handling; forward/backward and serialization checked. Research training pending |
| B: DAS plus residual U-Net | Implemented with residual encoder, spatial skip connections and waveform-context fusion; forward/backward and serialization checked. Research training pending |
| C: DAS-only | Residual-envelope imaging plus training-only scalar depth calibration implemented/tested; calibrated physical accuracy remains unverified |
| Ensemble uncertainty and calibration | Distinct-member inference and error/coverage diagnostics implemented; independent-seed frozen evaluation supported. Confidence calibration pending |
| Random, diagonal/edge fixed, RL adaptive | Random/fixed, masked reward-to-go actor/value RL, and a separate uncertainty-path comparator implemented/tested. No converged RL or physical performance claim |
| Equal-budget replay and fair architecture comparisons | Frozen plan/actual checkpoint lineage checks, matched initial actions/budgets, repeated random policies, dense references and parent-level summaries implemented/tested; research results pending |
| 128/256 outputs, extra decoder stage | Eight architecture/size/extra-block combinations pass single-step CPU debug checks against area-averaged 512 reference labels; trained matched ablations pending |
| Quantitative and regional metrics | Physical-unit and regional errors, overlap, centroid/missed detection, diameter/area/volume and ensemble diagnostics implemented/tested; final threshold/calibration pending |
| Curves, 2D and 2.5D plots, parameter counts/cost | Pending |
| Simulation-only versus physical adaptation | Source-model lineage, frozen unit scale and two-adaptation/three-holdout role workflow tested with fixtures; measured physical adaptation/evaluation pending |
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

## Waveform and neural implementation checks

The cache, scalar trace importer, public observation interface, nonlearning
policies, DAS and physical-unit metrics have regression coverage. External
CSV import rejects sampling/unit-header mismatches and records source hashes.
See data-contract.md. The point-scattering fixture is labeled plumbing-only.

Current suite: 58 tests with optional CPU PyTorch installed. Both neural
architectures pass hidden-NaN masking, permutation invariance, empty-state,
finite-gradient, unit/scaler, 128/256/extra-block and checkpoint checks.
`verify-models` separately ran exactly eight single-step debug updates at the
default model widths; every update had finite gradients and decreased its
fixture loss. These results do not compare generalization or establish an
accurate thickness inversion. See reconstruction-models.md and the compact
debug evidence for configuration counts, timings and resource samples.

The subset-training, exact checkpoint recovery, DAS scalar calibration,
ensemble adapters and frozen replay workflow now have implementation coverage.
Current suite: 75 tests. Exact Adam/subset/weight recovery and interruption
during validation are checked; physical role tests use two adaptation and
three held-out fixture parents. See experiment-workflow.md for commands and
remaining limits. No full research training cycle or real physical run occurred.

Still pending: randomized research-quality scenario/response generation and
physics approval, converged adaptive-policy experiments, uncertainty calibration, automatic matched
ablation/ensemble orchestration and plots, physical measurements, and a final
acceptance audit.

The runnable cache-to-replay pilot completed 18 matched three-method episodes
and four ensemble episodes in 16.56 seconds under resource monitoring. It
performed four total debug updates and explicitly labels its fixture metrics
as non-research results. Two additional tests exercise the actual CLI paths
for indexing, training/resume, scalar calibration, freezing and evaluation.
Current suite: 77 tests. See docs/evidence/workflow-debug.json and the
end-to-end section in experiment-workflow.md for recorded conditions.

## Adaptive selector checks

The RL policy learns from selected-record reconstruction improvements, with
labels restricted to the training/validation reward calculator. Reconstructor
weights remain frozen. Action masks, actor gradients, value-baseline detach,
exact policy checkpoint recovery, validation interruption, uncertainty-driven
heuristic changes and selector/model/holdout bindings pass regression checks.
Current suite: 89 tests. See adaptive-sensing.md for the objective, fixed-horizon
limit and distinction between RL and the path heuristic. Research-quality
policy training and physical efficiency remain unverified.

Policy CLI commands now support bounded training/resume, frozen selector
artifacts and the uncertainty-path comparator. Frozen plans also bind runtime
inference source files. A monitored two-update pilot completed 12 fixture
replay episodes across all four policy categories, with no repeats and common
dense predictions; the final resource/evidence record is in
docs/evidence/adaptive-debug.json. Current suite: 90 tests. These checks still
do not establish policy convergence, calibrated uncertainty or physical accuracy.

## Scenario and local forward-data implementation

The scenario planner and CAD mesher now feed an actual local elastic waveform
bank generator. Tests cover deterministic recipes/variants, CAD volume/mass
and geometric refinement, grouped transmitter solves, sample-aligned stable
steps, intact residuals, deterministic cached recovery/noise and rejection of
unsupported supports/unresolved geometry. Current suite: 99 tests with optional
Gmsh/PyTorch installed. These are pilot/geometry checks; generated banks are
not promoted to research-ready data. See scenario-generation.md.

