# Research readiness

This checklist follows the confirmed simulation and model-testing plan dated
November 3, 2026. Passing software checks permits further development; accepting
research training data also requires credible wave physics for the intended
plate, defect and transducer configuration.

| Requirement | Evidence available | Work still required |
| --- | --- | --- |
| Credible elastic forward solver | Independent operator, exact-field, dispersion and transient checks; the 40 mm intact P3 control and 20 mm paired CAD recess control pass fixed discrepancy screens; see highorder-waveforms.md and highorder-recesses.md | Full-size capacity and refinement, representative defect/frequency variation, independent continuum agreement and physical agreement |
| PZT transmission and voltage reception | Mechanical circular traction/displacement interfaces, including P1-P4 represented-field and native P2 checks | Bonded PZT/adhesive domains or a calibrated equivalent, drive and receiver electronics, measured material/input values |
| Reproducible synthetic specimens and waveforms | Scenario recipes, CAD recess geometry, bank fingerprints and masked replay workflows | Research-quality forward data, representative nuisance ranges and measured generation cost; existing pilot banks are not accepted |
| Specimen-level separation | Parent-level index/split checks and reserved-parent calibration workflows | Freeze research specimens and distribution assumptions before final evaluation |
| Direct waveform, DAS-assisted and DAS-only models | Implemented architectures/baseline; single-update model checks and matched-study pilot evidence | Meaningful validation on accepted data; debug results establish execution rather than model superiority |
| Random, fixed and adaptive selection | Masked replay, policy training and frozen equal-budget comparison workflows | Converged policy/model comparisons, repeated seeds and useful error-versus-budget curves |
| 128/256 maps and extra decoder depth | Common-grid losses and matched factorial study plans | Run planned comparisons on accepted data with paired observation exposure and fixed optimization budgets |
| Uncertainty calibration | Ensemble plans and held-out-parent interval fitting/assessment | Demonstrate useful coverage/width on representative validation data; freeze before final holdouts |
| Quantitative figures and acquisition costs | Physical-unit maps, curves, surfaces and replay cost tables | Publish scientific results from accepted data; measure physical switching/acquisition time separately |
| Resource-controlled verification | Process-tree guards with RAM, time and available GPU/temperature observations | CPU temperature has no supported reading on this PC; monitor research-scale pilots before larger runs |
| Physical adaptation and final holdouts | Import/frozen-evaluation interfaces and documented plate separation | Actual intact/control measurements, limited calibration and untouched physical holdout evaluation |

## Interpretation of numerical evidence

Earlier Elmer P2 studies used an under-integrated mass configuration. The
corrected rule agrees with exact quadratic mass and the independent high-order
assembly, but earlier dispersion/waveform/refinement measurements must be
regenerated before using them as evidence for the corrected formulation.
Their original failed screens remain historical records.

Even a successful intact-box discrepancy screen covers only its declared
geometry, source, receiver footprints, frequency and observation window.
Successive-grid differences are not a continuum error bound. An intact
40 mm control cannot establish 500 mm research-plate accuracy, recess
scattering or bonded-PZT voltage accuracy. No current result changes
`ready_for_training_data=false`.

## Bounded regression run

The matrix-free candidate now fits a full 500 mm intact P3 mesh/operator
measurement inside the 1 GiB guard (1,319,508 coefficients, 482.6 MiB sampled
combined RSS). It performs no propagation, and measured 6.5-second stiffness
actions leave affordable transient cost unproven. See matrixfree-reference.md.
Full-size wave accuracy, defect/PZT coupling and physical agreement remain open.
The first full-size acceleration probe reaches its 180-second guard after
fourteen completed iterations with no accepted state. All 243 native-enabled
regression tests pass without skips in 445.27 monitored seconds; checked
matrix-free recovery adds no new solves. The compact records are
evidence/matrixfree-debug.json and evidence/matrixfree-step-debug.json.

Use a new directory for each run. The wrapper retains its invocation, test log
and process-tree monitor result, and stops the test process tree on resource
limits. Individual training-related tests perform short debug updates; this
command does not launch a full training cycle.
Test temporary files are directed into the selected output directory.

```powershell
.\.venv\Scripts\python.exe -m examples.guarded_regression --output outputs/regression-check --elmer-home outputs/reference-tools/elmer-26.1/ElmerFEM-nogui-nompi-Windows-AMD64 --highorder
```

The Elmer path must identify an existing installation. Omit optional native
arguments only when those checks cannot run, and retain/report skipped tests.
Specific modules can be selected as positional arguments. Defaults are 300
seconds, 1,024 MiB combined RSS, 85% system RAM and at least 2,048 MiB available.
GPU usage and temperature are system-wide observations; CPU temperature is
recorded as unavailable when no supported sensor is exposed.
