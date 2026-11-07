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
.\.venv\Scripts\python.exe -m examples.guarded_highorder_reference --cad --plan outputs/highorder-recess-plan.json --settings configs/highorder-cad-refinement.json --output outputs/highorder-recess-refinement --monitor-output outputs/highorder-recess-monitor --cache-dir outputs/highorder-reference-groups
.\.venv\Scripts\python.exe -m traxis plot-reference-waveforms --sweep outputs/highorder-recess-refinement --records 0 3 --output outputs/highorder-recess-figures
```

The five-case protocol uses three mesh sizes at a common fine time step and
three time steps on the finest mesh. Mesh, footprints and operators must be
identical between temporal cases. Frozen criteria are 5% spatial and 1% time
relative L2 change at the worst receiver, with a 1e-18 m RMS floor. No fitting,
alignment or interpolation changes the raw comparisons. Exit status 2 means
a completed failed discrepancy screen; status 1 means an operational failure.

Absolute current-state waveforms can be dominated by direct arrivals. Their
agreement alone cannot establish convergence of the smaller defect-scattered
signal. Intact/current subtraction, full-size geometry, material and support
variation, calibrated PZT/electrical coupling and physical measurements remain
separate validation requirements. Every case retains
`ready_for_training_data=false`.
