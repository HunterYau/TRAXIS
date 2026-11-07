# High-order CAD recess references

The optional NGSolve mechanical worker supports straight-tetrahedron CAD
meshes with partial-depth circular backside recesses. `intact` omits the
declared recesses; `current` includes them and applies the declared coupling
drift. State, full scenario, mesh settings, Python source and Gmsh/native
runtime hashes belong to the cache identity. Neither state generates PZT
electrode voltages or accepted research training data.

## Bounded control

The committed control is a 20 x 20 x 3 mm plate with one seeded 8 mm-diameter,
1.5 mm-deep recess, eight circular mechanical footprints and a two-cycle
100 kHz Hann excitation. Its 128 samples are spaced by 0.5 microseconds.
This small problem exercises the defect interface before full-size work; it
does not replace the planned 500 mm research plate or its validation.

```powershell
.\.venv\Scripts\python.exe -m traxis plan-scenarios --config configs/highorder-recess-control.json --output outputs/highorder-recess-plan.json
.\.venv\Scripts\python.exe -m examples.screen_highorder_cad --plan outputs/highorder-recess-plan.json --settings configs/highorder-cad-refinement.json --output outputs/highorder-recess-topology
.\.venv\Scripts\python.exe -m traxis highorder-waveforms --plan outputs/highorder-recess-plan.json --settings configs/highorder-cad-waveform.json --state current --output outputs/highorder-recess-case --cache-dir outputs/highorder-reference-groups
```

Use fresh output directories. Install the optional high-order and Gmsh
dependencies in the project environment. Actual mesh topology and H1 counts
are checked before NGSolve allocation; the mesh size does not predict actual
degrees of freedom. Sparse export, factorization and process-tree guards
retain their existing limits. CPU temperature remains best-effort telemetry.

## Geometry and mechanical checks

Gmsh/OpenCASCADE subtracts the circular recess from the solid. The boundary is
then approximated by planar facets. Higher field order does not curve those
facets or remove geometric error. Each case records exact CAD and faceted
volumes and rejects removed-volume error above its declared limit. Geometry
and wave resolution both require refinement.

The top face remains flat. Analytic circular trace integration supplies
work-conjugate finite-area traction and averaged-displacement covectors.
Their area and represented monomial moments are checked before propagation.
All six rigid-body displacement fields must give negligible stiffness force,
and three uniform translations must recover density times the actual meshed
volume from the consistent mass. Solve residuals and postburst energy are
checked during the pulse integration. These checks establish mechanical
consistency, not continuum scattering accuracy.

## Mesh and time refinement

```powershell
.\.venv\Scripts\python.exe -m examples.guarded_highorder_reference --cad --plan outputs/highorder-recess-plan.json --settings configs/highorder-cad-refinement.json --output outputs/highorder-recess-refinement --monitor-output outputs/highorder-recess-monitor --cache-dir outputs/highorder-reference-groups --max-seconds 900
.\.venv\Scripts\python.exe -m traxis plot-reference-waveforms --sweep outputs/highorder-recess-refinement --records 0 3 --output outputs/highorder-recess-figures
.\.venv\Scripts\python.exe -m traxis plot-reference-waveforms --sweep outputs/highorder-recess-refinement/scattered --records 0 3 --output outputs/highorder-scattered-figures
```

The five-case protocol uses three mesh sizes at a common fine time step and
three time steps on the finest mesh. The committed configuration enables
paired scattering checks, so every level solves both intact and current
geometry for a total budget of ten groups. Mesh, footprints and operators
must be identical between temporal cases for each geometry. Frozen criteria are 5% spatial and 1% time
relative L2 change at the worst receiver, with a 1e-18 m RMS floor. No fitting,
alignment or interpolation changes the raw comparisons. Exit status 2 means
a completed failed discrepancy screen; status 1 means an operational failure.

Absolute current-state waveforms can be dominated by direct arrivals. The
paired study also compares the raw float64 current-minus-intact traces; both
absolute and difference signals must pass. It forbids coupling drift and
stochastic noise so those cannot masquerade as geometry-induced differences.
The report retains both states, every subtraction array and separate screen
results. Scattered plots have a separate checked view linked to the parent
report. Test fixtures explicitly reject a hidden difference-signal failure
even when direct arrivals agree. No amplitude normalization or fitted shift
is applied to either state before subtraction.

Full-size geometry, material and support variation, calibrated PZT/electrical
coupling and physical measurements remain separate validation requirements.
Every case retains
`ready_for_training_data=false`.

Baseline subtraction here includes the change in free-body motion caused by
removed mass under a net applied force. It is not an isolated propagating
scattered mode. A bonded PZT's internal actuation and voltage readout require
their own coupled or calibrated model; a rigid translation cannot be treated
as a measured piezoelectric voltage.

## Capacity preflight

`examples.screen_highorder_cad` builds only the six distinct meshes needed by
the paired study. It reports exact topological H1 counts and the conservative
assembly screen before any elastic operator or waveform is allocated. It
runs in a separate process with 60-second and 256 MiB defaults, plus system
memory and available temperature guards. Sparse fill and factor costs still
require a measured waveform pilot.

The initial 20,000-coefficient setting cannot hold the finest control meshes:
the intact/current 2 mm grids need 20,118/24,153 vector coefficients. The first
wave attempt was intentionally stopped after this was measured. Settings now
allow 30,000 coefficients while retaining the 1,024 MiB process-tree cap and
the original physics/error criteria. The topology preflight passes for the
updated setting and returns completed-screen status 2 for the old cap, with
zero wave solves in both checks. The 900-second wave-study wall guard reflects
the measured cost of ten groups, rather than relaxing spatial/time tolerances.

## Completed paired control, November 6–7, 2026

All five intact/current pairs completed with the original physical settings
and error thresholds. The damaged grids have 13,752 / 16,902 / 24,153 vector
coefficients; the intact grids have 11,307 / 14,073 / 20,118. Removed-volume
errors decrease from 3.9881% to 2.6859% to 1.9297%. The seeded recess center
is (10.04749, 9.93786) mm; radius is 4 mm and depth 1.5 mm.

| Signal/check | First pair, worst relative L2 | Final pair, worst relative L2 | Fixed limit | Screen |
| --- | --- | --- | --- | --- |
| Current waveform, spatial | 5.0318% | 2.3213% | 5% | Passed |
| Current waveform, temporal | 1.0272% | 0.25625% | 1% | Passed |
| Current-minus-intact, spatial | 10.1501% | 4.5183% | 5% | Passed |
| Current-minus-intact, temporal | 2.4216% | 0.60780% | 1% | Passed |

Observed time orders are 1.99934 for current waveforms and 1.99625 for the
subtraction. Every selected receiver exceeds the RMS floor. Maximum linear
residual is 2.59e-15 and maximum postburst relative energy range 1.28e-14.
These are discrepancy/algebraic checks, not continuum error bounds.

The capacity-adjusted run was interrupted after eight completed groups, with
no final study/monitor receipt. Its tool session was unavailable on resumption;
an explicit process check found no monitor or native worker alive. Source and
settings still matched the frozen plan. Recovery verified eight cached groups
and solved the two missing groups in 77.13 monitored seconds, peaking at
710.8 MiB combined RSS and 57.1% system RAM. GPU observations reached
1,412/8,192 MiB and 74 C from system activity; calculation used CPU only and
retained the 80 C GPU guard. CPU temperature was unavailable. The original
interruption's cause and total cost are not established. Its eight completed
case receipts cover at least 490.65 seconds and remain retained.

All five subtraction arrays are bitwise equal to saved current minus saved
intact traces. Both current and subtraction PNG/SVG figure sets were inspected.
The public `highorder-waveforms` command recovered the fine 62.5 ns current
group with identical arrays, one cache hit and zero new solves.

All 224 regression tests passed without skips with real Elmer/Gmsh/NGSolve
checks enabled. The run took 207.62 monitored seconds at 762.8 MiB combined
RSS, 69.4% system RAM and 50 C GPU temperature. An earlier run retained one
RAM-stop error: the existing Elmer cache test's 512 MiB envelope included the
already-imported test runner and worker. Its envelope is now 768 MiB under
the unchanged outer 1,024 MiB cap. RAM-stop messages now include measured
and allowed MiB. No numerical tolerance changed and no full training ran.

The compact source/runtime, worker, trace, plot, recovery and test record is
[highorder-cad-recess-debug.json](evidence/highorder-cad-recess-debug.json).
It also retains the original capacity stop, interrupted receipts, topology
screens and first failed regression log. All training-readiness flags remain
false. One small mechanical recess does not validate the full research plate,
other defect geometries, frequencies, bonded PZTs or physical measurements.

## Full-size feasibility

A separate geometry-only screen used a 500 x 500 x 3 mm plate with a 20 mm
diameter, 1.5 mm-deep recess and 5/3/2 mm candidate mesh sizes. It rejected
the configured local element/assembly limit at the coarsest size before
meshing, elastic assembly or any waveform solve. This is a pilot-limit
rejection, not a measured full-size memory/runtime benchmark. Its observation
window was not validated for full-size arrivals because no waves were solved.

The installed NGSolve 6.2.2606 build reports `USE_CUDA=False`. The presence of
an NVIDIA GPU does not make this installed reference GPU accelerated. The next
solver decision must address full-size cost and coupled PZT physics; see
[solver-assessment.md](solver-assessment.md).
