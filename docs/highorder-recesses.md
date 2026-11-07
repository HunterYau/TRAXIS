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
