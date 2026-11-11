# Research readiness

This checklist follows the confirmed simulation and model-testing plan dated
November 3, 2026. Passing software checks permits further development; accepting
research training data also requires credible wave physics for the intended
plate, defect and transducer configuration.

| Requirement | Evidence available | Work still required |
| --- | --- | --- |
| Credible elastic forward solver | Independent operators/fields, dispersion and transient checks; small P3 intact/recess controls; full-size enriched 500-microsecond window, second-order temporal convergence and CPU state checks | Full-size space/thickness refinement, representative defect/frequency variation and their time checks, continuum and physical agreement |
| PZT transmission and voltage reception | Mechanical circular interfaces; small synthetic coupled 3D/electrode/circuit reference with native assembly and independent exponential checks | Bonded PZT/adhesive domains or a calibrated equivalent, identified drive/receiver electronics, measured material/input values and coupled convergence |
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

The tensor investigation adds native/CUDA Q1-Q5 operators and analytic-profile
support across 2,592 sampled band cases, including folded modes. Its actual
5.42-million-coefficient full-plate operator fits the guard and agrees with
streamed CPU action. Physical circular footprints pass native checks; the
guarded wave workflow and independent saved-state checker pass a full-interval
small fixture. The first actual 500 mm tensor window completes 500 microseconds
and passes independent CPU state checks. Tensor convergence, conforming recesses
and calibrated PZT/physical agreement remain open. See lamb-profiles.md,
tensor-gpu-operators.md, tensor-wave-windows.md and their evidence records.

The full-size 1/2/4-layer thickness study completes all three 500-microsecond
windows and independent CPU checks, but fails its frozen 5% raw discrepancy
screen at every receiver: finest changes are 18.0010%, 9.90936% and 19.5625%.
Differences decrease, with in-plane cells fixed at 5 mm. This preserves the
spatial accuracy gate; it accepts no research data. See
layered-spatial-refinement.md and evidence/layered-thickness-debug.json.

The small coupled piezoelectric reference verifies constitutive signs,
electrode condensation and circuit energy on rectangular/skewed block
fixtures. All modal states are retained and independently audited. Its
synthetic material is not a hardware model, and its short circuit-check
interval is not a full plate wave window. See piezoelectric-reference.md
and evidence/piezoelectric-reference-debug.json. Data acceptance remains open.

The primary materials list proposes approximately 7 mm PZT discs, while the
current full-size mechanical control uses 6 mm disks. Its twelve-sensor ring
and eight-sensor edge-layout descriptions also differ. These remain proposed
hardware details, not calibrated numerical inputs; see hardware-assumptions.md.
The temporal study keeps its existing footprints fixed to isolate time error.

The full 500-microsecond temporal refinement now passes on one fixed intact
mesh at 100 kHz. Observed orders are 2.000018/2.000014; final raw trace changes
are 0.0144%/0.0096%. Source, runtime, mass and footprint coefficients match
exactly, and independent CPU final-state checks pass. This does not resolve
spatial/thickness error or validate another defect/frequency/hardware case.
See enriched-time-refinement.md and evidence/enriched-time-debug.json.

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

A subsequent float64 GPU element backend completes the same full-size first
equation in 12.08 solve seconds / 78.80 monitored seconds, with true residual
9.53e-12 and peak combined RSS 568.20 MiB. A separate native CPU action check
confirms that residual. All 247 integrated native CPU/GPU tests pass without
skips in 477.00 monitored seconds; the later standalone CPU checker and seven
input-contract checks were verified separately. See gpu-element-reference.md
and evidence/gpu-element-debug.json. One first step leaves full-window,
mesh/time, defect/frequency and PZT/physical acceptance open.

Optional element additive Schwarz preconditioning reduces this first solve
to 70 iterations / 7.03 seconds, with current native CPU residual agreement.
Extra factorization raises total monitored time to 95.17 seconds and system
GPU memory to 5,264 MiB. All 250 integrated native CPU/GPU tests pass without
skips in 498.92 monitored seconds at 894.28 MiB combined RSS. See
gpu-element-preconditioning.md and evidence/gpu-schwarz-debug.json. Longer
propagation and its cost have not yet been demonstrated.

Device-resident CG and revised kernel indexing reduce the observed first
solve to 5.33 seconds with the same 70 iterations and independent native CPU
residual agreement. Full-size CG transfers two vectors to the GPU and one
back; Newmark state handling remains on the CPU. All 253 integrated native
CPU/GPU tests pass without skips in 528.19 monitored seconds at 875.42 MiB
combined RSS. See gpu-resident-cg.md and evidence/gpu-resident-debug.json.
This is still one full-size step, with full-window and physical acceptance open.

A separate 15-node positive-mass enriched candidate passes native/P4 operator
and discrete temporal checks. Its full-size native space and topology-order
positive mass fit the guard, with 1,173,621 vector coefficients. All 29 targeted
element/integrator checks pass without skips; the last complete integrated
suite at that cutoff was the earlier 253-test result. See enriched-tetrahedra.md
and evidence/enriched-tetrahedron-debug.json. Those element/capacity receipts
alone do not prove full-size mapping, explicit cost or waveform acceptance.

The later factorized enriched GPU method verifies the full native/topology map
and completes the unchanged 500 mm / 500-microsecond intact mechanical window:
31,469 steps, 216.08 propagation / 266.09 monitored seconds, 644.25 MiB sampled
combined RSS and GPU peak 77 C. Independent streamed CPU final-state energies
agree to 3.08e-14 relative. A separate 72-case structured-cell study reaches
0.0251% maximum finest Lamb-frequency error, while coarse bending-mode errors
reach 3.67% and one-layer lateral refinement is nonmonotone. This is not a
full-size wave-convergence or PZT/physical acceptance. See
enriched-gpu-waveforms.md and evidence/enriched-gpu-wave-debug.json.

The final complete integrated run passes all 268 tests without skips, with real
Elmer, high-order and GPU references enabled, in 616.504 test / 622.304 monitored
seconds. Sampled combined RSS peaks at 876.81 MiB, system RAM at 57.6% and GPU
temperature at 65 C. Short model/policy debug updates are included; no full
training cycle ran. This supersedes the earlier complete-suite counts above.

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
