# Implementation and evidence checklist

The source of requirements is [the consolidated research plan](https://docs.google.com/document/d/1YdcENA3Y00U-8t2Yn2oDq9uq08JuIf8LWrp96weIiNI/edit),
read 2026-11-03 and rechecked 2026-11-11. A local snapshot is in `docs/research-plan-source.txt`.

Status means implemented and verified only when evidence is named here.
Full training and physical experiments are explicitly outside the short
debug runs authorized for this session; provide working code and honest
validation rather than inventing those outcomes.

| Requirement | Current state / acceptance evidence needed |
| --- | --- |
| Solver recommendation | COMSOL preferred if licensed; portable Elmer independent assembly, P2 dispersion and finite-plate trials run. P2 spatial screen failed; coupled PZT benchmark remains open |
| 3D elastic propagation and backside loss | Local linear-tet FEM and independent Elmer P2 mechanical reference implemented; algebraic/dynamic/dispersion checks run. Spatial and physical acceptance remain open |
| Independent higher-order assembly | Optional NGSolve P1-P4 checks pass for operators, exact energies, circular footprint moments and coarse reciprocal pulses; P2 agrees with fresh Elmer transients. Spatial/time convergence and physical acceptance remain open |
| PZT electromechanics, supports, noise/coupling variation | Seeded nuisance variation and a small synthetic coupled 3D/electrode/circuit reference verified; bonded PZT/adhesive/support modeling, identified electronics and measured calibration still needed |
| Intact arrival/dispersion and mesh/time convergence | Rayleigh-Lamb, local FEM and independent Elmer P2 Bloch checks implemented. P1/P2 finite-plate time screens passed, mesh screens failed; PZT/physical validation still needed |
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
| Frequent commits on project branches | research/elastic-tomography, research/finite-plate-validation and research/elmer-reference; tested implementation and evidence increments committed |

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

## Quadratic independent guided modes

Elmer P2 operators now support a separate infinite-plate dispersion study
against independent Rayleigh-Lamb traction roots. Correct periodic face
relations, free interior P2 values, sagittal restriction and conjugate
Hermitian reduction have checks. Three in-plane refinements hold thickness
fixed; three thickness refinements hold in-plane spacing fixed. Five shared
cases per target produce 40 bounded exports over 100/200 kHz and 1/3 mm stock.
The default plan and tolerances are frozen before assembly. See elmer-waveguide.md.

Every finest-grid frequency error was below 0.0677%, using 243 full / 72
condensed degrees of freedom. The overall predeclared screen **failed**:
100 kHz S0 in 1 mm stock had thickness errors increasing from 0.001735% to
0.001843%. All in-plane and seven other thickness sequences decreased.
The small nonmonotone component was not reclassified as a pass or hidden by
a tolerance change. Physical finite-plate/PZT validation remains open.

The study took 21.58 seconds with sampled combined RSS 109.9 MiB, system RAM
61.7%, GPU memory 921/8,192 MiB and GPU temperature 47 C. CPU temperature was
unavailable; all calculation used CPU and no training ran. Checked PNG/SVG
refinement figures were visually inspected. Compact evidence is in
docs/evidence/elmer-waveguide-debug.json. These results justify investigating
high-order reference waveforms, not promoting existing coarse local banks.

All 156 regression tests pass with real P1/P2 solver checks enabled. The full
suite took 60.25 seconds with sampled combined RSS 779.7 MiB, system RAM
64.9%, GPU memory 921/8,192 MiB and GPU temperature 47 C. A separate bounded
nine-case diagnostic reduced the thin-S0 error to 0.0001242% and 0.000009380%
at 32/64 in-plane cells and four thickness cells. The corresponding thickness
sequences decreased, supporting an in-plane dispersion interaction. This
follow-up preserves the original failed criterion and data-readiness flag.

## Independent transient loads and integration

The verify-elmer-transient command now freezes and checks two tiny free-body
meshes, uniform volume/nodal load equivalence and a discrete elastic mode at
three time refinements each. SaveScalars columns are read by checked names
and requested times. The explicitly chosen Bossak alpha=0 method gives
second-order average-acceleration Newmark. Analytic zero-start responses are
checked against an independent adaptive ODE solve, including resonance and
small-time limits. See elmer-transient.md.

All 18 transient runs passed. Uniform-load finest errors were 0.02036%;
elastic-mode errors were 0.3777%, with observed time orders 1.9969–2.0007.
Equivalent volume/nodal traces differed by at most 1.04e-12 relative L2.
The P2 load has 18 negative nodal coefficients; these were retained and
verified against the positive uniform physical volume load.

The pilot took 31.24 seconds with sampled combined RSS 124.3 MiB, system RAM
61.8%, GPU memory 921/8,192 MiB and GPU temperature 47 C. CPU temperature was
unavailable; no GPU computation or training ran. Evidence is in
docs/evidence/elmer-transient-debug.json. These are temporal/interface checks
on fixed coarse systems. Finite-area apertures, spatially converged finite
plates/recesses and PZT/physical validation remain open.

All 161 regression tests pass with the optional real reference checks enabled.
The full suite took 90.95 seconds with sampled monitor/test-tree RSS 781.3 MiB,
system RAM 64.8%, GPU memory 921/8,192 MiB and GPU temperature 47 C. Probe
tests reject wrong times, missing/duplicate named columns, and analytic
responses agree with an independent adaptive ODE solver at/near resonance.

## Quadratic mechanical source and receiver apertures

Signed P2 circular integrals now preserve area, centroid and second moments.
Independent slice/Gaussian quadrature, analytic moments and quadratic fields
check the integration across remeshing and node permutations. Native P2 mesh
loading checks body/type/counts, straight edge ordering and conforming shared
midpoints. Curved quadratic geometry is rejected. See quadratic-apertures.md.

The three-run native pilot checks swapped aperture loads/probes and a
truncated Hann source against independent scalar kinematics. Reciprocal
traces differ by 5.39e-13 relative L2, and the source interface by 1.27e-12.
Area errors are 2.22e-16 relative, centroid errors below 1.54e-18 m, and
second-moment errors below 6.36e-21 m². Negative nodal weights were retained.

The pilot took 3.78 seconds with sampled combined RSS 116.2 MiB, system RAM
62.4%, GPU memory 921/8,192 MiB and GPU temperature 47 C. CPU temperature was
unavailable; no training or GPU computation ran. Evidence is in
docs/evidence/quadratic-aperture-debug.json. All 168 regression tests passed;
the corrected all-permutation aperture subset also passed 12 tests. Existing
P1 banks remain pilot-only, and spatially converged finite-plate/PZT/physical
validation remains open.

The 168-test full run took 95.02 seconds with sampled monitor/test-tree RSS
783.2 MiB, system RAM 65.8%, GPU memory 921/8,192 MiB and GPU temperature
47 C. The pilot's nominal 1 N force scale multiplies the unnormalized Hann
sine; its sampled pulse maximum was 0.8602 N. CPU temperature was unavailable.

## Bounded quadratic finite-plate references

The reference-waveforms command now assembles independent Elmer P2 stiffness
and consistent mass once and integrates them in an isolated sparse worker.
Native-transient comparisons, independent coupled adaptive ODE tests,
energy conservation, permutation/archive checks and checked group recovery
cover the new path. No PZT domains or training-data acceptance are implied.
See elmer-waveforms.md and docs/evidence/elmer-waveform-debug.json.

The native 75-node comparison differs by 1.74e-12 relative L2. The final-source
40 x 40 x 3 mm intact control used 3,777 P2 nodes, 1,928 tetrahedra and seven
receivers, taking 9.69 seconds at 302.2 MiB sampled combined RSS. Its maximum
linear residual is 5.98e-16 and postburst energy range 9.46e-15 relative.
These algebraic checks leave finite-plate spatial/time convergence open.

All 177 regression tests passed in 104.72 seconds with optional native checks
enabled. Sampled combined RSS peaked at 783.5 MiB, system RAM at 66.2%, GPU
memory at 921/8,192 MiB and GPU temperature at 47 C. CPU temperature was
unavailable. No training ran. Source snapshots and input/kernel/runtime
hashes bind the pilot; all solver/source changes remain inside this project.

## Quadratic finite-plate refinement

The five-case independent P2 mesh/time study completed for one intact-plate
transmitter and seven receivers. Target mesh sizes were 3/2/1.5 mm at a
common 31.25 ns step; temporal cases used 125/62.5/31.25 ns on the finest
mesh, with identical mesh/aperture hashes. Source/kernel/runtime and saved
trace checks guard the comparisons. See elmer-plate-refinement.md.

The time screen passed at 0.2205% maximum finest-pair change and order
1.99917. The spatial screen failed at 30.84%, following 12.27% on the first
pair. Thresholds remain 5% space/1% time, and ready_for_training_data=false.
Separate through-thickness/in-plane refinement and PZT/physical validation
remain open; the cause of the spatial discrepancy is not established.

The completed run took 327.70 seconds at 640.4 MiB sampled combined RSS and
79.4% system RAM. GPU observations peaked at 1,690/8,192 MiB and 51 C;
CPU temperature was unavailable. The 15,478-node system completed after
tested sparse factor/diagnostic memory changes, within the original 768 MiB
cap. Earlier time/RSS interruptions, source snapshots and their costs are
retained in docs/evidence/elmer-plate-refinement-debug.json. No training ran.

All 183 regression tests passed with native checks enabled in 112.02 seconds,
at 782.6 MiB sampled combined RSS and 78.3% system RAM. GPU observations were
1,531/8,192 MiB and 48 C; CPU temperature was unavailable. Five-case CLI cache
recovery took 7.49 seconds with zero new solves, identical saved arrays and
the expected failed-screen exit code 2. Refinement figures were visually checked.

## Independent in-plane, thickness and time checks

A seven-case structured intact-box study now separates both spatial axes
and time, sharing the finest case. The reference rejects recess geometries
and verifies exact box volume, screened P2 node counts, physical apertures,
source/kernel/runtime identities and saved comparison arrays. The CAD path
continues to support conforming recesses. See elmer-directional-refinement.md.

In-plane changes decreased from 57.14% to 12.83%; thickness changes from
41.60% to 8.01%. Both fail the predeclared 5% screen. Time refinement passed
at 0.2205% and order 1.99909. These results leave spatial accuracy unresolved
in both directions, with all data-readiness flags false.

The initial 768 MiB attempt stopped on the finest grid after two completed
groups. A measured-capacity recovery preserved those groups and all physics,
raised the RSS cap to 1,280 MiB and added outer guards of 85% system RAM and
2,048 MiB minimum available memory. It completed in 486.33 seconds under
the outer monitor at 1,009.9 MiB sampled combined RSS and 76.5% system RAM.
GPU observations peaked at 1,831/8,192 MiB and 48 C; CPU temperature was
unavailable. Earlier stops, costs and independent algebraic memory probes
are retained in docs/evidence/elmer-directional-refinement-debug.json.

All 189 regression tests passed with native checks enabled in 114.88 seconds,
at 785.1 MiB sampled test-tree RSS and 76.6% system RAM. GPU observations were
1,378/8,192 MiB and 49 C; CPU temperature was unavailable. The committed
guarded example recovered all seven groups with identical saved arrays,
zero new solves and the correct failed-screen exit code 2 in 5.53 seconds.
Raw and difference PNG/SVG figures were visually checked. No full training ran.

## Independent high-order assembly and mass correction

NGSolve 6.2.2606 is installed only in the project virtual environment, with
four hash-pinned Windows/Python 3.12 wheels and unchanged NumPy/SciPy. The
isolated P1-P4 assembly check covers rectangular and affine-skewed blocks,
exact monomial/mixed-strain energies, six rigid modes and Ritz refinement.
See highorder-reference.md for commands and scope.

The first comparison exposed under-integration in the previous Elmer P2
mass configuration: a 4.656% nodal-matrix error against exact barycentric
integrals. Exports and native transients now explicitly request the higher
rule, reducing error below 2.86e-15 on both blocks. The waveform cache binds
elements.def. All earlier P2 dispersion/transient/refinement numbers are
historical and require regeneration for the corrected formulation. Their
provisional flags and failed screens are preserved; see elmer-mass-quadrature.md.

All eight corrected degree/geometry cases pass, with polynomial energy
errors below 5e-14 and P2 operator agreement with NGSolve below 2.63e-14.
The native pilot took 3.72 seconds at 291.3 MiB sampled combined RSS.
A corrected P2 waveform control completed in 8.37 seconds at 256.5 MiB;
it remains unsuitable for research training pending spatial/physical checks.

All 197 regression tests passed in 134.49 seconds with both native libraries
enabled. Sampled combined RSS peaked at 771.2 MiB, system RAM at 72.6%, GPU
memory at 1,386/8,192 MiB and GPU temperature at 45 C. CPU temperature was
unavailable. The earlier full-suite guard stop and original failed mass
comparison are retained in docs/evidence/highorder-reference-debug.json.
No full training ran.

## Higher-order circular footprints

The native P1-P4 coefficient basis now has analytic circular mechanical
traction/displacement covectors. All 48 cases pass across baseline,
renumbered and refined meshes, including full, half and quarter disks.
Exact represented monomial averages, oblique directions and independent
slice quadrature check the interface. P2 covectors agree with the nodal
interface within 5.74e-16 relative error; coarse pulse traces agree with
fresh corrected native Elmer transients within 1.62e-12. All degrees pass
reciprocity and postburst energy checks on the fixed coarse mesh.

The retained pilot took 11.42 seconds at 256.4 MiB sampled combined RSS.
All 205 full-suite tests passed in 147.96 seconds with both native solvers
enabled; five focused tests passed after the final CLI-only adjustment.
Full-suite peaks were 772.9 MiB combined RSS, 73.4% system RAM and 45 C GPU
temperature. CPU temperature was unavailable. Source snapshots, native
inputs, footprint covectors, sparse operators, traces and receipts remain
hash-linked in evidence/highorder-aperture-debug.json. See
highorder-apertures.md for commands and fixed tolerances.

This establishes discrete mechanical interfaces. Finite-plate spatial/time
convergence, the production higher-order waveform pipeline, bonded-PZT
physics and physical calibration remain open. Research training readiness
remains false, and no full training ran.

## Higher-order intact-plate waveform refinement, November 6

The seven-case P4 run completed on the original 40 x 40 x 3 mm intact control.
Thickness and time screens passed at 0.2093% and 0.2217% maximum final-pair
relative L2 change. In-plane refinement failed at 18.888%, with the original
5% criterion retained. Its outer monitor recorded 305.48 seconds, 628.5 MiB
combined RSS and 72.6% system RAM. A fresh-output recovery reused all seven
groups with identical arrays, zero new solves and the correct exit status 2.

The existing P3 7/8/9-cell configuration then completed all seven cases under
the same resource and error limits. Final-pair changes were 3.4578% in-plane,
0.5939% through thickness and 0.2240% in time. All three declared screens
passed, with observed temporal order 1.99907. The finest case has 23,520
vector coefficients. The finest P4/P3 traces differ by at most 2.1195% in
a supplemental comparison; no continuum bound is claimed.

The P3 outer monitor recorded 408.61 seconds, 684.4 MiB combined RSS, 73.7%
system RAM and at least 4,262 MiB available. GPU observations peaked at
1,007/8,192 MiB and 45 C. All 213 regression tests passed without skips with
both native solvers enabled, in 210.31 monitored seconds at 771.0 MiB combined
RSS. Five resource/child-termination tests passed after the regression wrapper
was updated to keep temporary files inside its project output directory.
Dependency checks passed. CPU temperature was unavailable; no GPU calculation
or full training ran. Raw/difference figures were inspected.

See highorder-waveforms.md and evidence/highorder-waveform-refinement-debug.json
for frozen settings, source/runtime and artifact hashes, checked arrays,
resource receipts and the retained failed P4 run. The passing P3 screen applies
only to this intact mechanical control. Full-size plates, recess scattering,
bonded-PZT voltage output and physical calibration remain open. All research
training-data readiness flags remain false.

## CAD recess and paired-scattering checks, November 6–7

The high-order worker now accepts explicit intact/current CAD geometry and
binds Gmsh kernels, physical state and mesh settings into each job. It checks
removed volume, circular footprint moments, rigid-body forces and translation
mass before propagation. A topology-only worker counts actual H1 coefficients
without elastic assembly or time integration. The seeded 20 mm control's fine
intact/current grids require 20,118/24,153 coefficients, so its declared
capacity is now 30,000 under the unchanged 1,024 MiB outer RAM guard.

The ten-group paired study passes its frozen 5% spatial/1% temporal screens:
final current-waveform changes are 2.3213%/0.25625%; raw current-minus-intact
changes are 4.5183%/0.60780%. Observed time orders are 1.99934 and 1.99625.
Geometry error decreases to 1.9297% removed volume. Maximum linear residual
is 2.59e-15 and postburst relative energy range 1.28e-14. These differences
include rigid-motion changes and do not establish isolated PZT scattering.

After an interruption with no final receipt, actual process inspection found
the monitor and workers absent. Recovery verified unchanged source/settings,
reused eight groups and solved only the remaining two. It took 77.13 seconds
at 710.8 MiB combined RSS and 57.1% system RAM. GPU observations reached
1,412/8,192 MiB and 74 C from system activity; calculation used CPU. CPU
temperature was unavailable. Cause/total cost of the interruption are unknown;
the eight completed receipts and earlier intentional capacity stop are retained.

All 224 regression tests pass without skips with real native solvers, in
207.62 monitored seconds at 762.8 MiB RSS and 69.4% system RAM. The original
512 MiB test-envelope stop remains recorded; the test now allows 768 MiB
inside the 1,024 MiB suite guard. No physics tolerance changed. Current and
subtraction figures were inspected; subtraction arrays are exact saved-state
differences. The public command recovers the finest current group with zero
new solves. No full training ran.

The 500 mm geometry-only screen rejects the current pilot element/assembly
budget before meshing. NGSolve's installed build reports CUDA disabled. Full
plate scaling, coupled/calibrated PZT voltage responses and physical validation
remain required. See highorder-recesses.md and
evidence/highorder-cad-recess-debug.json for the exact scope and receipts.
No research training-data acceptance is claimed.

## Matrix-free full-size capacity, November 7

The new matrix-free candidate retains the checked vector H1 elastic and
consistent-mass forms. A tiny P1-P4 rectangular/skewed all-column and direct
Newmark comparison passed eight cases before recovery hardening. Exact
effective-operator diagonals addressed an initial P4 CG iteration failure;
one 180-second attempt still hit its wall guard. Complete case receipts now
check shapes, expanded size, exact times, counts and recomputed saved metrics.
All 239 full-suite tests now pass without skips with real native solvers in
475.49 monitored seconds at 776.95 MiB combined RSS, 56.8% system RAM and
77 C system GPU temperature. The eight current-source native comparisons
completed in 287.25 seconds and recovery reused all cases with zero new
computations. Raw comparison evidence was retained before test cleanup.

An actual 500 x 500 x 3 mm intact 5 mm CAD mesh and P3 matrix-free operators
completed in 64.15 monitored seconds with 1,319,508 vector coefficients.
The two stiffness/mass actions cost about 6.5/2.7 seconds each. Translation
mass, null force and work reciprocity pass. Peak sampled combined RSS was
482.6 MiB, system RAM 53.9%, minimum available RAM 7,476 MiB, and GPU system
temperature 77 C. Calculation used CPU, with CPU temperature unavailable.
No sparse assembly, time propagation or training ran. This establishes
bounded capacity, with transient cost and full-size wave/PZT accuracy open.
See matrixfree-reference.md, evidence/matrixfree-debug.json and
outputs/matrixfree-fullplate-capacity-20261107. Failed CG/wall-limit attempts
remain recorded. No full training ran and no research bank is accepted.

## Full-size first-step cost, November 7

The shared strict CG solve now supports saved iteration timing/count/norm
progress; the original Newmark integrator uses that same checked path. Larger
mechanical footprints require explicit scalar/element/workspace limits and
screen before allocating extraction arrays. Twelve focused checks pass,
including an actual small native first-step comparison to sparse acceleration
and direct Newmark. The first direct-reference fixture exceeded its existing
2,000-coefficient cap; that rejection now happens before solving.

An actual first Newmark acceleration equation on the retained 500 mm P3
mesh, 3 mm-radius force disk and 125 ns step stopped at the 180-second guard
after fourteen completed CG iterations. Last progress was at 170.47 seconds;
each iteration cost roughly 9-10 seconds. There is no final true residual
or completed state. Peak sampled combined RSS was 494.48 MiB, system RAM
45.4%, minimum available memory 8,860 MiB and system GPU temperature 57 C.
CPU temperature was unavailable; the run used CPU and performed no training.

This is a cost/resource stop, not a numerical-accuracy failure or usable
waveform. Affordable full-window propagation remains unproven. The modified
tree's full native regression now passes all 243 tests without skips in
445.27 monitored seconds at 775.54 MiB combined RSS, 46.0% system RAM and
43 C system GPU temperature. A retained 1,671-coefficient P2 native step
agrees with sparse acceleration to 3.04e-11 and direct Newmark's probe to
1.55e-15, with true residual 7.98e-12. No full training ran. See
matrixfree-step-cost.md, evidence/matrixfree-step-debug.json and
outputs/matrixfree-fullplate-first-step-20261107 for retained evidence.

## Float64 GPU element reference, November 7

Five hash-pinned optional CUDA/CuPy wheels were installed in the project
virtual environment, preserving NumPy 2.5.4/SciPy 1.18.1. A guarded float64
kernel matched its NumPy reference and dependency checks passed. No global
toolkit, compiler or driver change occurred. NGSolve's installed build still
has CUDA disabled; this backend streams its checked native local matrices
and applies them separately through CuPy/NVRTC.

All eight P1-P4 rectangular/skewed GPU cases pass every-column K/M/fused-A,
diagonal, buffer/block and 32-step sparse-direct comparisons. Worst waveform
change is 2.75e-11 and true residual 1.001e-11. The initial run took 16.16
monitored worker seconds at 427.6 MiB combined RSS and 53 C GPU temperature.
Eleven focused GPU/CPU driver/integrator checks pass. The full integrated
native/GPU regression passes all 247 tests without skips in 477.00 monitored
seconds at 830.89 MiB combined RSS, 47.1% system RAM and 46 C GPU temperature.

The unchanged full-size P3 first acceleration equation now completes in
242 iterations / 12.08 solve seconds, 78.80 monitored worker seconds total,
with true residual 9.53e-12 and work/energy error 6.66e-16. Sampled combined
RSS was 568.20 MiB, system RAM 45.3%, minimum available memory 8,876.9 MiB,
GPU system memory 3,135/8,192 MiB and GPU temperature 63 C. CPU temperature
was unavailable. Local matrices occupy 2,242.65 MiB on the device, with no
global sparse assembly. No full training ran.

A separate native CPU action check of the saved full-size GPU acceleration
gives residual 9.527939665e-12 versus GPU 9.527926869e-12, exact source-covector
agreement and energy/probe differences below 4.45e-16. It completes in 36.91
monitored seconds at 404.15 MiB combined RSS without CPU CG or sparse assembly.
Seven saved-input checks pass; the standalone checker was verified after the
247-test run. Compact identities and receipts are in evidence/gpu-element-debug.json.

This passes one first-step cost/consistency check, not a full-size observation
window or research-bank acceptance. Full-size space/time and defect/frequency
accuracy, coupled/calibrated PZT voltage and physical agreement remain open.
See gpu-element-reference.md and outputs/gpu-fullplate-first-step-20261107.

## Element preconditioning, November 7

An optional symmetric, multiplicity-weighted element additive Schwarz inverse
passes all-column local-solve reference, symmetry/positive-factor and 32-step
direct comparisons on rectangular/affine P1-P4 cases. Fourteen focused tests
pass. It preserves the full equations and checked residual targets. Its local
inverse/weight storage adds 2,021.74 MiB on the full-size plate.

The unchanged full-size first equation converges in 70 iterations / 7.032
solve seconds with residual 7.43908e-12. Native CPU actions confirm 7.43907e-12.
Total monitored time increases to 95.17 seconds because setup costs more;
the previous diagonal solve took 12.08 seconds / 78.80 total. Peak sampled
combined RSS is 586.98 MiB, system RAM 48.6%, available memory at least
8,339.71 MiB, system GPU memory 5,264/8,192 MiB and GPU temperature 64 C.
CPU temperature remains unavailable. The diagonal baseline remains default.
Full-window propagation, full-size refinement, defect/PZT/physical acceptance
and research training remain unfinished. See gpu-element-preconditioning.md.

The integrated native CPU/GPU regression passes all 250 tests without skips
in 495.937 test seconds / 498.918 monitored seconds. Peak sampled combined
RSS is 894.28 MiB, system RAM 49.4%, available RAM at least 8,205.86 MiB and
system GPU temperature 42 C. Current-source array/receipt checks and compact
evidence are retained in evidence/gpu-schwarz-debug.json. The saved acceleration
differs from the earlier diagonal GPU result by 7.65e-11 relative L2; independent
current native CPU actions also pass. No full training ran.

## GPU-resident CG, November 7

The optional device-resident CG solver keeps iteration coefficient vectors
and reductions on the GPU, with scalar synchronization and one accepted-state
download. Recurrence convergence and a fresh original-equation residual are
both required. The kernels also avoid integer division in their inner component
loops. Storage preflight reserves sixteen vectors; the original guards remain.

All eight rectangular/affine P1-P4 direct-reference cases pass, along with
warm/zero-state, transfer-count and nonconvergence checks. Seventeen focused
tests pass; the native contract probe was then isolated and checked separately
to release its CUDA context before later worker tests.

The unchanged 500 mm P3 first equation completes in 70 iterations / 5.326 solve
seconds, 93.573 monitored seconds total. GPU and native CPU residuals are
7.438959e-12 and 7.438951e-12. CG uploads two vectors, downloads one, and reads
213 scalar reductions. Sampled combined RSS is 599.82 MiB, system RAM 48.2%,
available RAM at least 8,403.89 MiB, GPU memory 5,352/8,192 MiB and temperature
66 C. CPU temperature remains unavailable. Newmark states still use CPU arrays.

All 253 integrated native CPU/GPU tests pass without skips in 525.159 test
seconds / 528.187 monitored seconds, at 875.42 MiB combined RSS, 49.9% system
RAM and 43 C GPU temperature. Compact evidence is in
evidence/gpu-resident-debug.json; see gpu-resident-cg.md. Full-window cost,
full-size refinement, defect/PZT/physical acceptance and research training
remain open. No full training ran.

## Enriched explicit candidate, November 7-8

A separate 15-node quadratic-plus-bubbles tetrahedron has positive nodal mass
and exact comparison stiffness/consistent-mass integrals. Independent basis,
quadrature/strain, permutation, continuity and native enriched/P4 embedding
checks pass. The largest matrix discrepancy is 4.10e-14. A retained coarse
temporal ratio miss of 4.239 is followed by 64/128/256-step convergence over
the same interval, with unchanged criteria and ratios approaching four.

The installed native mass-lumping docstring describes 2D support, but actual
3D probes return the required 15-node tetrahedron. The unchanged full-size
mesh has 391,207 scalar / 1,173,621 vector coefficients and finite positive
nodal mass totaling 2.025 kg. All ten degree-zero-to-two mass moments pass
to 1.51e-13. This space/mass-only run took 4.241 monitored seconds at 303.84
MiB combined RSS; no stiffness, CUDA allocation or time propagation ran.
Full-size native/topology coefficient permutation remains unverified.

All 29 targeted enriched/explicit/implicit checks pass without skips with
NGSolve, Gmsh and Elmer enabled in 22.515 monitored seconds at 312.02 MiB
combined RSS. The last complete integrated suite is the earlier 253-test
run. CPU temperature remains unavailable; no full training ran. See
enriched-tetrahedra.md and evidence/enriched-tetrahedron-debug.json.

This candidate needs a mesh-specific explicit stability limit, scalable GPU
operators, cubic surface traction/probes, full-window cost and spatial/defect/
frequency validation before acceptance. Coupled/calibrated PZT voltage and
physical agreement remain open; the existing consistent-mass path is unchanged.

## Full enriched GPU window and dispersion, November 8

The factorized float64 stiffness retains all enriched displacement coefficients
without dense local matrices. Native maps, mass, exact local CPU actions and
explicit stability bounds pass on the unchanged full-size mesh. Cubic circular
traces handle the enriched face bubbles; lower-degree fits are rejected.

The full 500-microsecond intact window completes at 15.8887 ns increments:
31,469 steps, 216.077 propagation / 266.092 monitored seconds. The postburst
modified-energy relative range is 6.47e-16. Independent CPU final-state energies
agree to 3.08e-14 and probes to 3.96e-16 relative. Sampled peaks are 644.25 MiB
combined RSS, 57.2% system RAM, 3,503 MiB system-wide VRAM and 77 C GPU temperature.

Seventy-two structured periodic Lamb cases vary in-plane and thickness
resolution independently. Finest frequency errors stay below 0.0251%, but
coarse bending errors reach 3.67%; one thickness cell retains a nonmonotone
in-plane sequence. These controls do not establish full-size wave accuracy.
The comparison report's stiffness/waveform variable-reuse bug is fixed, with
saved-matrix re-evaluation in the native test. Unperformed checks use null
values. See enriched-gpu-waveforms.md and evidence/enriched-gpu-wave-debug.json.

Full-size space/thickness/time and recess/frequency studies, PZT/electrical
calibration, physical agreement, accepted banks and meaningful model/policy
experiments remain open. No full training ran.

The final complete regression passes all 268 tests without skips, with Elmer,
NGSolve/Gmsh and GPU checks enabled. Test time is 616.504 seconds; monitored
time 622.304 seconds. Peak sampled combined RSS is 876.81 MiB, system RAM 57.6%,
available RAM at least 6,876.92 MiB and GPU observations 3,139 MiB / 65 C.
The compact evidence includes final test-source hashes, numerical archives,
independent CPU checks and resource receipts. The earlier 253-test count is
historical; selected-suite counts are not added to the complete 268.

## Complete-window time convergence, November 8

The unchanged full-size mesh/source/receivers complete 31,469, 62,938 and
125,876 explicit steps over the same 500 microseconds. Exact coarse sample
matching avoids interpolation. Raw per-receiver difference ratios are
4.000050/4.000038 and orders 2.000018/2.000014; final changes are
0.0143737%/0.00955326%. Both fixed screens pass. CPU-reconstructed mass, final
stiffness/energy and receiver checks also pass for every level.

The quarter run takes 982.005 propagation / 1,036.204 monitored seconds,
including 37.233 seconds of verified-worker cooling pauses. It reaches but
does not exceed the original 80 C GPU limit, with 650.68 MiB peak combined
RSS and 57.4% system RAM. Numerical states, forcing/sample times and original
guards are unchanged by host scheduling. New tools have focused tests; the
last complete integrated suite remains the earlier 268. No training ran.

See enriched-time-refinement.md and evidence/enriched-time-debug.json. Full-size
space/thickness refinement, representative defects/frequencies, PZT/electrical
calibration and physical agreement remain open. hardware-assumptions.md records
the proposed 7 mm transducers and unresolved sensor layout separately from the
6 mm disks used in this numerical control.

## Full-size layered thickness result and coupled reference

On November 9 the 500 mm, 100 kHz, three-receiver 1/2/4-layer study completes
all three 500-microsecond windows at the same 65,536 increments. Each native
coefficient map, GPU/CPU operator check and independent final-state check
passes. The largest case has 3,315,627 vector coefficients, takes 1,041.389
propagation / 1,189.113 monitored seconds, and peaks at 982.02 MiB combined
RSS and 75 C. All pacers complete with no pauses.

The declared thickness screen fails: raw finest changes are 18.0010%,
9.90936% and 19.5625%, all above 5%, despite decreasing differences. The
complete unchanged window and receiver set are retained. See
layered-spatial-refinement.md and evidence/layered-thickness-debug.json.
The cost and accuracy of further thickness/in-plane resolution remain open.

A separate small synthetic coupled piezoelectric reference passes independent
native 3D operator, affine-energy, electrode-condensation and open/short
circuit checks on rectangular/skewed blocks. Six passive/driven circuit
comparisons against matrix exponentials show second-order ratios 3.99557 to
3.99896. Three focused invariant tests pass; the last full integrated suite
remains 268 tests. Raw modal states/matrices are independently audited.
See piezoelectric-reference.md and evidence/piezoelectric-reference-debug.json.
This is a constitutive/port building block, not bonded PZT plate or physical
validation. No accepted bank, full research training or model superiority
result is implied by either milestone.

## Higher-order tensor-element investigation

The isolated tensor-element branch verifies Q1-Q4 hexahedral operators with
explicit exact/GLL stiffness rules and separately checked positive GLL mass.
Eight connected-element rectangular/skewed native comparisons pass at a
maximum 4.26e-14 relative block discrepancy. Six focused quadrature, action,
energy, rigid-mode, shared-mass, limit and physical-parity tests pass without
skips. A first complex-root dtype failure is retained and corrected by an
explicit checked-real conversion.

The 432-case periodic design study separates A0/S0 by midplane parity. Its
360 within-zone branches are identified; 72 out-of-zone comparisons are
withheld. All sixteen declared finest cases pass the 0.05% frequency design
screen. At nominal 100 kHz/3 mm with 5 mm cells and one thickness cell, Q4 A0
error is 0.00299911%, versus 0.265193% for Q3. This does not establish the full
pulse-band, finite-plate or defect/PZT result. Full-size tensor counts are
forecasts only; capacity, acceleration and full-window convergence remain
unmeasured. See tensor-elements.md and evidence/tensor-reference-debug.json.
The prior complete suite remains 268 tests and no research training ran.

## Analytic profile support and full-size tensor GPU operator

The frozen 2,592-case 5-250 kHz profile study captures 99.9951% of nominal
force spectral energy. Modes are selected by 99% analytic-profile mass norm,
with the worst selected frequency error screened at 0.05%. This assesses 234
outside-first-zone cases, including 150 multi-mode supports. The 5 mm Q4
carrier-only mesh fails the sampled band screen; 3.333 mm passes for uniform
3 mm stock, while uniform 1 mm stock needs the tested 2.5 mm spacing.
All 96 finest cases pass. See lamb-profiles.md and its raw audit.

The new matrix-free tensor CUDA action passes every native stiffness column
on eight Q1-Q4 rectangular/skewed cases and two additional multi-axis sharing
cases. A small native eigenmode shows error ratios 4.001095/4.000274 and
modified-energy range below 5.08e-16. Ten selected CPU and two enabled native
tests pass without skips; these are not a new complete-suite total.

The actual 500 mm intact Q4 grid has 150-by-150-by-1 elements and 5,418,015
vector coefficients. GPU/streamed-CPU full-field discrepancy is 9.87e-16,
mass is 2.025 kg and box-moment error is below 1.14e-13. The worker completes
in 10.6254 seconds at 631.20 MiB combined RSS, 50.7% system RAM and 52 C.
Five isolated device actions take 4.488-5.386 ms. It performs zero full-plate
propagation steps; physical patch integration, complete-window accuracy/cost,
recess/PZT modeling and accepted data remain open. See tensor-gpu-operators.md
and evidence/tensor-gpu-debug.json. No research training ran.

## Physical tensor ports and full-window workflow preflight

The reference and CUDA action extend through Q5. Connected skewed Q5 native
comparisons check every stiffness column, GLL mass, exact stiffness and
consistent mass. Physical 3 mm radius circular covectors integrate the
piecewise top trace, retain signed weights and check analytic area and
represented moments. Five independent native-shape fixtures pass below
1.14e-12 relative against a fixed 1e-9 screen. The current-source reference,
CUDA, footprint and cooling-selection group passes 24 tests without skips.

Three drivers freeze the complete p/time study, execute guarded tensor
windows and independently check saved states with local-scatter CPU mass
and streamed stiffness. Raw ZIP/NPY shape/header bounds are checked before
array loading. Nine selected protocol/comparison/state tests pass; these
overlap the earlier checks and are not a new full-suite total.

The complete 500-microsecond small smoke window runs 32,768 steps with
1,215 vector coefficients in 13.235 propagation seconds. CPU final energies
agree within 2.23e-16 relative and mass within 2.53e-16. The measured peak is
438.26 MiB combined RSS, 49.9% system RAM and 1,047 MiB / 53 C GPU. CPU
temperature is unavailable. See tensor-wave-windows.md and
evidence/tensor-wave-preflight-debug.json. Full-size tensor windows and
convergence remain pending, with no research training/data acceptance.

## First full-size tensor mechanical observation window

Q4-N32768 in the frozen tensor study completes the unchanged 500 mm plate
and 500-microsecond interval: 5,418,015 vector coefficients, 32,768 steps,
243.828 propagation / 259.031 monitored seconds. The full action agrees
with streamed CPU stiffness to 9.87e-16 relative. Independent local-scatter
CPU mass, final physical/modified energies and probes agree within
4.14e-16 relative. Raw postburst modified-energy range is 2.33e-15.
All circular coefficients reproduce exactly, with analytic area/moments
below 5.45e-15.

Peak combined RSS is 919.96 MiB, system RAM 53.1%, minimum available RAM
7,607.75 MiB and GPU observation 1,551 MiB / 74 C. The attached exact-worker
cooling monitor terminates normally without a pause. CPU temperature is
unavailable. This worker finished November 9; the CPU audit and figures were
checked November 10. See tensor-wave-windows.md and
evidence/tensor-fullplate-window-debug.json. Four remaining frozen cases,
representative recess/frequency studies and physical calibration are still
required. No research training or accepted data bank results from one window.

## Half-step tensor window and two-level diagnostic

Q4-N65536 completes all 65,536 steps at 7.62939453125 ns over the same full
500-microsecond interval. Propagation takes 494.814 seconds and the monitored
worker 510.183 seconds. Independent CPU final energies agree at the saved
floating-point precision, mass within 4.14e-16 and final probes within 1.63e-16.
Combined RSS peaks at 920.70 MiB, system RAM at 57.5%, GPU memory at
1,449 MiB and temperature at 75 C. Cooling completes without a pause;
CPU temperature remains unavailable.

Exact mass, physical patch and common time-sample checks pass across the two
Q4 levels. Raw receiver changes are 0.0549479%, 0.0364132% and 0.0504256%.
This is not a three-level convergence assessment. The 131,072-step Q4 window
and Q3/Q5 spatial windows remain. See tensor-wave-windows.md and the
tensor-half-window-debug.json / tensor-first-time-pair-debug.json evidence
records. No source, window, receiver, criterion or physical scale was changed.

## Complete coarse tensor spatial level

Q3-N65536 completes the same full-size 500-microsecond experiment with
2,440,812 vector coefficients. Propagation takes 224.531 seconds; monitored
worker time is 233.804 seconds. Native/CPU final-state checks agree within
3.96e-16 and the full GPU/CPU action within 4.50e-16 relative. Peak RSS is
634.79 MiB, system RAM 51.1%, and GPU observations 1,185 MiB / 74 C.
The cooling monitor completes without suspension. Its raw states, CPU check
and audited figures are bound by evidence/tensor-q3-window-debug.json.
The Q5 spatial level and 131,072-step Q4 level remain required. Completion
of this coarse window does not erase its retained periodic-band failures.

## Passing complete tensor spatial comparison

Q5-N65536 completes the unchanged 500-microsecond experiment with
10,152,018 vector coefficients in 973.007 propagation / 998.247 monitored
seconds. Independent CPU mass, energies and probes agree within 4.19e-16;
the full operator within 1.06e-15. Peak RSS is 1,370.93 MiB, system RAM
56.6%, and sampled GPU memory/temperature 1,881 MiB / 77 C. Its optional
cooling monitor records a Windows progress-file replacement failure and
successful same-worker reattachment. The simulation guards remain active,
and no numerical state/input is restarted or changed.

The complete Q3/Q4/Q5 spatial screen and independent raw-data audit pass:
finest changes are 0.117856%, 0.100770% and 0.109101%, with decreasing
differences at every receiver. These are successive-discretization changes
under the declared intact mechanical controls, not continuum-error bounds
or recess/PZT/physical acceptance. See tensor-wave-windows.md and
evidence/tensor-q5-window-debug.json / tensor-spatial-refinement-debug.json.
The finest temporal level is still required at this cutoff.

## Nonaffine tensor reference for the geometry extension

A separate CPU/native example verifies two connected hexahedra under a
degree-two polynomial deformation, through Q3/Q4/Q5 solution spaces.
Whole-matrix native stiffness/mass discrepancies are below 8e-15. Exact
volume, affine energy, physical nodes and exactly six rigid modes pass a
separate raw-array audit. The 1,188-coefficient maximum fixture completes
the three-case worker in 3.561 monitored seconds at 353.90 MiB RSS.

Two initial failures are retained: a grid/layout API mismatch and native
deformation's requirement for VectorH1 rather than a compound scalar space.
See mapped-tensor-reference.md and evidence/mapped-tensor-reference-debug.json.
This is an independent geometric reference, with no mapped CUDA solver,
circular recess mesh, transient or data acceptance yet. The active frozen
intact wave source inventory remains unchanged.

## Complete tensor time refinement and frozen-source replay

Q4-N131072 completes the unchanged full 500-microsecond interval in
1,009.584 propagation / 1,025.123 monitored seconds, including eleven
cooling pauses totalling 11.718 seconds. GPU temperature peaks at 78 C;
RSS at 924.16 MiB. Independent CPU final-state checks pass within 6.67e-16.
The complete three-level raw time comparison and separate audit pass:
orders are 2.000022/2.000009/2.000006 and finest changes
0.0137368%/0.00910324%/0.0126064%. Exact discrete mass, physical covectors
and shared time samples match across levels.

All five frozen forward cases and CPU checks are complete. Spatial and
temporal screens pass for this intact mechanical experiment. See
tensor-wave-windows.md and evidence/tensor-quarter-window-debug.json /
tensor-time-refinement-debug.json. Representative recess/frequency and
PZT/physical gates remain open; ready_for_training_data stays false.

Guarded exact-source replay reproduces the spatial assessment from the
retained archive, validating inventory, extraction paths and every source
hash. It preserves strict historical identities after later code changes.
See examples/replay_tensor_assessment.py and
evidence/tensor-source-replay-debug.json. This replays postprocessing,
not forward physics or full training.

## Bounded Windows progress publication recovery

The shared atomic writer retries brief Windows replacement denials with a
finite 0.455-second backoff budget. A bounded shared JSON reader accompanies
the fix. The first reader-only held-open test failed and remains retained;
sharing flags alone were insufficient on this machine. Seven focused and
13 integration tests pass. A fresh complete small GPU window and independent
CPU check agree with the old control, and both historical full-plate
assessments reproduce exactly from the archived numerical source.

See progress-log-recovery.md and evidence/progress-log-recovery-debug.json.
The workflow fixture runs two one-step CPU debug updates; no full or research
training cycle ran. This is I/O reliability work, not new physics acceptance.

## Conforming recess geometry and separate circle resolution

The geometry extension now builds explicit shared hexahedral connectivity,
recess walls/floors and paired intact material. The 90-case screen covers all
planned diameters/depths and geometry degrees one through five. Ten small
native geometry comparisons and an independent coordinate-derivative audit
pass. The six new geometry tests and existing tensor/storage checks pass all
19 tests without skips. An initial hand-derived node-count error and native
report-construction failure remain retained.

The area result alone hid a larger radial error in a coarse cubic circle.
A subsequent frozen screen requires both area and radius accuracy, retains
the one- and two-element failures and selects four cubic elements per quarter.
All 18 planned paired domains and two small native comparisons then pass.
Peak process-tree RSS is 397.24 MiB; observed GPU temperature reaches 51 C.
These are CPU geometry checks; CPU temperature is unavailable.

See conforming-recess-geometry.md, recess-shape-refinement.md and their compact
evidence receipts. Geometry arrays, native coordinates/Jacobians, failed
attempts, source snapshots and independently audited CSV/PNG/PDF figures are
retained. Mapped elastic stiffness/mass, stability, physical disk integration,
GPU capacity and full paired wave/space/time studies remain open. No research
training, bank acceptance or PZT/physical validation follows from geometry.

## Mapped recess elastic action, positive mass and stability

The mapped CPU action evaluates variable geometric Jacobians and shared
connectivity without storing dense element stiffness. A positive diagonal
energy majorant supplies a conservative explicit bound for each geometry.
The final focused group passes 25 tests without skips, including a separate
spectral check for positive/negative Lamé lambda and allocation rejection.

All eight small native cases pass: Q3/Q4/Q5 paired intact/recess on the
selected circle geometry, plus two small connected Q3 full-matrix/spectrum
references. A separate strain-matrix/native-Jacobian audit verifies 511
elements and 200,352 columns, all local mass/nullity/majorant checks and
global native applications. Maximum independent column discrepancy is
3.51e-13. Native jobs peak at 400.31 MiB RSS and GPU observations at 51 C;
the jobs are CPU-only and CPU temperature is unavailable.

See mapped-recess-operators.md and evidence/mapped-recess-operators-debug.json.
No transient, GPU propagation, physical disk integration, full-size defect,
representative thin-stock/frequency or PZT/physical acceptance follows from
this operator checkpoint. No research training or bank generation ran.

## Revised stock and PZT mounting randomization

The current physical target is 3.175 mm aluminum with removable clamp-mounted
PZTs. The explicit research configuration now randomizes positions within
0.5 mm, effective circular contact radius by 10%, coupling by 10%, additional
gain drift, force scale and noise. It enables independent baseline/current
remounts. Ranges are provisional sensitivity settings, not measured clamp
tolerances. See current-hardware.md and pzt-domain-randomization.md.

Actual ports drive local and native waveform integration/cache keys, while
inference retains nominal coordinates. Remount variants preserve plate,
material, parent and split. New mounting plans/tasks use schema 2; nominal
legacy fingerprints remain unchanged. The final 50-test integration and a
saved small 3.175 mm waveform sensitivity control pass, including native pulse
checks and an independent recipe/port audit. See the compact domain-randomization
evidence receipt. No model training or accepted research-bank generation ran.

Historical 3 mm references remain frozen. Current full-size mesh/time/frequency
validation, measured clamp/contact/preload/couplant response and physical PZT
agreement remain open. This randomization layer does not model clamp mechanics.

## Mapped recess CUDA action at 3.175 mm

The explicit curved-mesh GPU operator now uses per-GLL-point inverse geometry
and shared-node gather/scatter. It retains positive mass and the CPU diagonal
energy stability bound and works with the device-resident Verlet integrator.
Allocation and finite-state checks run before CUDA execution.

Eight fresh native cases pass: paired Q3/Q4/Q5 selected geometry plus small
Q3 intact/recess matrices. Native checks cover 511 elements and 200,352 local
columns; 3,336 global GPU columns are compared directly. Independent replay
of retained native element scatter agrees with GPU forces within 3.32e-13;
maximum small-matrix column error is 4.07e-14. Free and driven modal checks
at 32/64/128 steps show second-order displacement/velocity errors and modified
energy agreement. These synthetic intervals are about 55-57 ns, not plate-wave
recording windows.

The final focused group passes 18 tests without skips in 9.531 seconds.
Native/GPU jobs peak at 524.47 MiB RSS, system RAM 52.7% and GPU observations
1,072 MiB / 53 C; tests observe 1,074 MiB / 54 C. CPU temperature is unavailable.
See mapped-recess-gpu.md and evidence/mapped-recess-gpu-debug.json. Earlier
3 mm references stay frozen. Full-size mapped capacity, physical/randomized
surface ports, complete wave convergence and measured clamp/PZT agreement
remain open. No model training or accepted waveform-bank generation ran.

## Mapped circular ports and complete small pulses

Curved flat-top surface integration now resolves each state's actual PZT
centers and contact radii. Analytic integration in the ruled coordinate and
adaptive disk-clipped integration in the other preserve signed work-conjugate
load/probe vectors. State/geometry/allocation checks remain explicit; coupling
gains stay separate from unit-force footprints.

Six Q3/Q4/Q5 native intact/recess runs compare 24 disk covectors using
independent native shapes and Gaussian integration. Maximum discrepancy is
3.20e-14. Five Q4 pulse controls complete 31.5 us each, totaling 48,384 steps.
Reciprocity error is 2.68e-15 and the halved-step trace change is 3.34e-6
relative. A remounted intact source changes the waveform; this is sensitivity
of the small mechanical control, not measured clamp variability.

The final focused group passes 23 tests without skips in 18.853 seconds.
An independent retained-array audit verifies vectors, pulse comparisons,
energy and final CPU/GPU forces; the final PNG/PDF is visually checked.
Native/pulse jobs peak at 546.55 MiB RSS, 54.0% system RAM and GPU observations
1,076 MiB / 58 C. CPU temperature is unavailable. See mapped-surface-ports.md
and evidence/mapped-surface-pulse-debug.json.

The rechecked research-plan snapshot is unchanged; the user's later 3.175 mm
stock and removable-clamp instructions define current hardware. Full-size
mapped capacity, spatial/frequency/complete-window convergence, identified
PZT/contact traction and physical validation remain open. No model training
or accepted bank generation ran.

## Full-size compact recess capacity

The compact planar/extruded builder preserves the reference geometry and
topology with a different global numbering and int32 connectivity. Small
degree-one through degree-five permutation checks and native Q3 all-column,
mass and spectrum checks pass. CPU mapped action can declare a smaller batch
budget while existing references retain the default 32-vector limit.

Full 500 by 500 by 3.175 mm Q4 recess/intact cases build at 4,664,847 and
5,543,451 displacement coefficients. CPU/GPU field discrepancies are below
1.20e-15; analytic mass/centroid and physical disk-port checks pass. A separate
whole-cell polynomial audit gives positive floating-point Jacobian bounds
for all 46,592 cells. The final focused suite passes 34 tests without skips
in 26.911 seconds. Retained failures cover a test-helper typo and Windows
Path handling in the first capacity preflight; neither changes tolerances.

Measured peak process-tree RSS is 1,254.00 MiB, system RAM 59.4%, minimum
available RAM 6,596.85 MiB and GPU observations 1,534 MiB / 52 C. CPU
temperature is unavailable. The 0.234375–10.800764 mm edge-size range creates
a restrictive stability bound while exterior wave resolution is unverified.
A 500 us operator-only extrapolation is 29.33/35.78 minutes, excluding all
integration/sampling/cooling overhead. No propagation ran in these capacity
jobs. Rebalancing the mesh and measuring full-size wave convergence remain
the next numerical work. See full-size-recess-capacity.md and its compact
evidence receipt. Training and research-bank readiness remain false.

