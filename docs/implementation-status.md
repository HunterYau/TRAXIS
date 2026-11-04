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
| PZT electromechanics, supports, noise/coupling variation | Seeded material, equivalent force/gain/drift/noise variation implemented; PZT/adhesive/electrical/support models and measured calibration still needed |
| Intact arrival/dispersion and mesh/time convergence | Rayleigh-Lamb reference and 3D FEM Bloch dispersion convergence implemented; finite-plate waveform/arrival convergence, resolved recess geometry and physical traces still needed |
| Randomized specimens and exact depth labels | Seeded geometry/nuisance recipes, intact controls, nine size/depth combinations, clearances, parent-preserving variants and area-averaged labels implemented/tested; validated research dataset pending |
| CAD-conforming geometry and elastic bank generation | Gmsh/OpenCASCADE recess mesh with volume/refinement checks and bounded grouped 3D elastic solves/cache implemented; banks remain plumbing_only pending wave/PZT/physical validation |
| Waveform bank indexed by specimen, TX, RX, frequency/settings | NPZ schema, integrity/bounds, anchored scalar CSV import, actual local elastic pilot generation and checked TX-frequency cache implemented; validated research dataset pending |
| Specimen-level splits and hidden observations | Fingerprinted lazy dataset, role-aware subset training/adaptation and checkpoint-bound held-out evaluation enforce parent separation; masked replay tested |
| A: waveform CNN, attention, spatial decoder | Implemented with geometry/sampling/amplitude context, partial masks and empty-state handling; forward/backward and serialization checked. Research training pending |
| B: DAS plus residual U-Net | Implemented with residual encoder, spatial skip connections and waveform-context fusion; forward/backward and serialization checked. Research training pending |
| C: DAS-only | Residual-envelope imaging plus training-only scalar depth calibration implemented/tested; calibrated physical accuracy remains unverified |
| Ensemble uncertainty and calibration | Frozen independent-member inference and reserved-parent split-conformal interval workflow implemented/tested; useful coverage/precision and exchangeability remain unverified |
| Random, diagonal/edge fixed, RL adaptive | Random/fixed, masked reward-to-go actor/value RL, and a separate uncertainty-path comparator implemented/tested. No converged RL or physical performance claim |
| Equal-budget replay and fair architecture comparisons | Matched factorial study planning/advances, actual training-draw checks, frozen pair/ensemble plans, equal-budget replay and parent summaries implemented/tested; research results pending |
| 128/256 outputs, extra decoder stage | Eight configurations/two paired trials pass bounded study training/validation on the common 512 reference; converged matched ablations pending |
| Quantitative and regional metrics | Physical-unit and regional errors, overlap, centroid/missed detection, diameter/area/volume and ensemble diagnostics implemented/tested; final threshold/calibration pending |
| Curves, 2D and 2.5D plots, parameter counts/cost | Headless PNG/SVG figures, physical-unit maps/surfaces and parent/compute tables implemented; research results pending |
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
physics approval, converged adaptive-policy experiments, useful independent uncertainty calibration,
matched research ablation results, physical measurements, and a final
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

Recipe planning, individual local-bank generation and external reference-task
export now have CLI coverage. Exports include a tested SI CAD script, label/time
anchors and mapped scalar-probe jobs; they do not run a licensed solver or
invent missing PZT material/circuit data. Current suite: 103 tests. A monitored
CAD-to-bank pilot records the improved geometry and actual elastic solves in
docs/evidence/forward-debug.json; its readiness flag remains false.

The final pilot completed in 2.74 seconds with sampled process RSS of 120.4 MiB,
system RAM at 77.2% and GPU telemetry at 1,669/8,192 MiB and 47 C. It used no
GPU arrays. Recess-volume error fell from 49.2% on the coarse structured mesh
to 0.422% on the finest CAD benchmark mesh; this measures geometry only.
Seven intact and fourteen defective transmitter/frequency solves generated
the small banks; replay recovery used fourteen cached groups and zero new
solves. The 1,000 study specimens are unsolved recipes, not a generated dataset.

## Scientific reporting checks

Completed schema-2 evaluations now bind geometry, predictor configurations,
parameter counts, platform/thread settings and saved-map hashes. The headless
reporter checks selected maps against their saved metrics and produces
physical-unit PNG/SVG curves, loss/error/spread maps, 2.5D surfaces and
parent/compute CSV tables. Distinct learned-policy artifacts stay separate
in summaries; only random seeds pool as repeats. See scientific-reporting.md.

Current suite: 110 tests pass, including changed-artifact/protocol rejection,
map/metric disagreement and plotting without importing PyTorch. Dependency
checks pass. Existing debug checkpoints produced 18 matched A/B/C and six
ensemble replay episodes with no further training. Nine figure pairs were
rendered and inspected; the final render took 12.17 seconds with sampled RSS
178.0 MiB, system RAM 80.5%, GPU usage 1,666/8,192 MiB at 47 C. CPU temperature
was unavailable. Replay processes stayed below 341 MiB sampled RSS. These
are software fixtures and single timing observations, not research results.
The compact record is docs/evidence/reporting-debug.json. Calibrated confidence,
matched research ablation results and physical validation remain open.

## Reserved-parent calibration checks

Separate `uncertainty_calibration` parents are excluded from model/policy
fitting and selection. Interval settings bind to a frozen replay plan before
calibration acquisition/reconstruction. The post-hoc score takes each parent's
worst normalized depth error over reference cells, saved budgets and policies.
The corrected order statistic, physical clipping, zero-spread floor, small-n
full-range result, changed-protocol rejection and parent reuse protections are
tested. See uncertainty-calibration.md for the exact scope and assumptions.

All 121 tests and dependency checks pass. A monitored pilot performed zero
training updates, 18 reserved-fixture and four held-out replay episodes in
19.88 seconds. Sampled RSS reached 352.6 MiB, system RAM 81.4%, GPU memory
1,669/8,192 MiB and GPU temperature 48 C. It used CPU inference; CPU temperature
was unavailable. The dedicated freeze/fit/assess CLI sequence was also checked
on the saved replay without further inference. Evidence is committed in
docs/evidence/calibration-debug.json.

The pilot does **not** establish usable calibration: intervals averaged
2.990 mm in 3 mm stock, with one of two held-out fixture parents jointly
covered versus a nominal 90% target. The workflow's software `passed` flag is
separate from `empirical_coverage_meets_nominal=false`. Settings were not tuned
to improve those final outcomes. Independent exchangeable calibration data,
research-quality models and useful coverage/precision remain unverified.

The revised study configuration plans 603 train, 156 validation, 92 uncertainty
calibration and 149 final-test parents for seed 2026. These are still 1,000
unsolved recipes, not a generated or approved training dataset.

## Common-reference training objective

Training, validation and scalar DAS fitting now retain the original reference
labels and defect weights across output resolutions. Native prediction cells
are repeated/averaged onto that grid for the loss. Analytic gradient and
single-cell-defect checks prevent a lower-resolution target from hiding small
defects. DAS fitting also normalizes each image's weighted objective before
combining images. Historical native-grid loss values are not comparable.

All 124 tests pass. Eight new single-step checks, one per architecture/size/
extra-block configuration at the default widths, pass on the shared 512 grid.
Sampled process RSS reached 427.1 MiB, system RAM 79.6%, GPU usage
1,668/8,192 MiB at 47 C; CPU temperature was unavailable. Evidence is in
docs/evidence/common-grid-debug.json. No full training cycle was run.

## Matched study orchestration

The factorial planner and bounded job runner share training/validation parents,
a training-only signal scale, the common reference loss and paired
initialization/subset seeds. Actual draw hash prefixes must match within each
trial. Fixed update budgets, unique run identities, exact recovery, final
validation with zero new updates, completed-artifact checks and a cross-process
execution lock have regression coverage. Planning requires intact and damaged
training parents and does not read reserved or final-test labels.
See matched-studies.md for commands and scope.

All 133 tests and dependency checks pass. The final monitored pilot trained
16 neural instances for exactly one update each, fitted four scalar DAS
baselines and froze 24 single-factor pair comparisons plus eight ensemble
plans. It replayed one predeclared pair per factor and one ensemble on
validation fixtures (16 episodes), checking equal acquired actions. Remaining
plans were not executed. These runs verify software behavior, not trained
architecture quality or physical accuracy.

The pilot completed in 19.34 seconds with sampled RSS of 448.2 MiB, system RAM
81.8%, GPU usage 1,670/8,192 MiB and GPU temperature 48 C. Training/inference
used CPU only; CPU temperature was unavailable. The compact record is
docs/evidence/matched-study-debug.json. No full research training cycle or
final physical evaluation was run. Research-quality data, finite-plate/PZT
validation, useful calibration and converged model/policy comparisons remain
open.

## Circular mechanical footprint integration

The local FEM now integrates each top-face P1 shape function over its actual
disk intersection, replacing centroid-based inclusion. This keeps source/probe
area and first-moment center fixed under remeshing while preserving mechanical
reciprocity. The method and its limits are documented in mechanical-patches.md.

All 138 tests pass with no skips. Exact full/half/quarter-disk moments, vertex
permutations, SI scaling, affine fields and independent adaptive slice integrals
cover the new geometry integration. The independent quadrature needed explicit
circle/edge breakpoints to resolve a thin cap; no tolerance was relaxed.
All 19 existing mechanical numerical gates also pass.

The CAD-to-bank pilot completed in 2.07 seconds with sampled RSS 124.6 MiB,
system RAM 79.8%, GPU usage 1,666/8,192 MiB and GPU temperature 47 C. Its full
patch-area error was at most 2.3e-16 relative and centroid error below 1e-17 m.
CPU temperature was unavailable. See docs/evidence/circular-patch-debug.json.
The banks still declare `ready_for_training_data=false`: exact footprints do
not establish resolved bending waves, PZT coupling or physical agreement.

## Finite-plate mesh and time refinement

The new check-waveforms command freezes three mesh sizes, three geometric
time-step refinements and error thresholds before solving. The spatial cases
share one fine time grid; the temporal cases share the finest mesh. Checked
float64 raw probes prevent bank rounding or fitted alignment from hiding
differences. Recovery reuses verified transmitter/frequency groups in a fresh
attempt directory. Plotting verifies report/plan/trace integrity and exports
physical-unit PNG/SVG overlays and differences. See waveform-validation.md.

All 144 tests pass. The completed 35-solve intact mechanical pilot took 43.08
seconds, with sampled RSS 122.9 MiB, system RAM 61.6%, GPU usage 921/8,192 MiB
and GPU temperature 44 C. CPU temperature was unavailable; propagation used
CPU. An earlier Delaunay run hit its time guard because of a measured sliver;
quality-only probes led to HXT/Netgen before the final waveform comparisons.

The temporal screen passed: the final pair differed at most 0.0209%, with
observed order 2.00025. The spatial screen **failed**: the final mesh pair
differed at most 145.70%. Some elements still span the full thickness, and
the shortest screened fundamental wavelength is only 2.94 maximum edges.
Tolerances and traces were not adjusted to manufacture a pass. The compact
record is docs/evidence/waveform-refinement-debug.json; the readable trace
plots were inspected. No training was performed. Spatially resolved research
data, independent elastic reference and PZT/physical validation remain open.

## Independent Elmer operator reference

A project-local portable Elmer release now provides an independent assembly
check without a commercial license. The verify-elmer command exports tiny
free rectangular and affine-skewed 3D P1 meshes, restores checked matrix node
ordering and compares stiffness, exact consistent mass and row-summed lumped
mass. Six rigid modes and a positive seventh are checked. Source/module and
input/output hashes, the actual solver banner and external-process telemetry
are retained. See elmer-reference.md.

Both real reference cases passed: maximum stiffness error 1.38e-14, consistent
mass error 3.70e-15 and lumped-mass error 3.04e-15 in relative norm. The two-case
command took 0.53 seconds with sampled combined RSS 105.0 MiB, system RAM
61.3%, GPU usage 921/8,192 MiB and GPU temperature 47 C. CPU temperature was
unavailable; no GPU calculation or training was performed. The distribution
is named release 26.1, while its executable reports 9.0 compiled 2026-01-20;
the distinction is recorded. No global installation/settings changed.

All 150 regression tests pass with the optional real solver test enabled.
The full run took 59.49 seconds with sampled monitor/test-tree RSS 779.1 MiB,
system RAM 64.8%, GPU memory 921/8,192 MiB and GPU temperature 47 C. Parser
tests cover nonidentity node ordering and malformed exports; an actual
sleeping external process verifies termination/reaping on a time guard.

The compact evidence is docs/evidence/elmer-reference-debug.json. Agreement
on identical coarse elements verifies assembly, not continuum wave accuracy.
The finite-plate spatial screen remains failed. Converged transient reference
waveforms, actual PZT/adhesive/electrical inputs and intact physical validation
are still required before research data generation or model comparisons.

