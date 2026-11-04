# Bounded quadratic mechanical waveform references

`reference-waveforms` runs one transmitter/frequency group on a CAD plate
with independent Elmer P2 stiffness and consistent mass. A separate monitored
Python worker integrates those exported operators with average-acceleration
Newmark. This makes larger mechanical checks practical without repeating
native element assembly at every time step. It performs no training and
always records `ready_for_training_data=false`.

```powershell
.\.venv\Scripts\python.exe -m traxis plan-scenarios --config configs/waveform-control.json --output outputs/waveform-control-plan.json
.\.venv\Scripts\python.exe -m traxis reference-waveforms --plan outputs/waveform-control-plan.json --settings configs/elmer-waveform-control.json --elmer-home outputs/reference-tools/elmer-26.1/ElmerFEM-nogui-nompi-Windows-AMD64 --output outputs/elmer-p2-control --cache-dir outputs/elmer-waveform-groups --max-seconds 60 --max-rss-mib 512
```

Use an existing scenario plan or create it once with the first command. Use
a fresh output directory for each reference run. `--transmitter`,
`--frequency-khz` and `--state intact|current` choose the group and geometry.
Only eligible records in the scenario's directed/reciprocal action catalog
are exported. This single group is not a complete waveform bank.

## Mechanical and numerical definitions

The plate is free, isotropic and lossless. Gmsh creates a conforming P1 CAD
mesh, including backside recesses for `current`; ElmerGrid inserts straight
edge midpoints to produce ten-node tetrahedra. This changes the field basis,
not the planar approximation of curved CAD surfaces. Curved P2 geometry is
rejected by the mesh reader. Node/element/matrix estimates and recess-volume
error are screened before assembly. These screens do not certify wave accuracy.

Signed [circular P2 apertures](quadratic-apertures.md) integrate the normal
traction and average normal displacement. Area, centroid and second moments
are checked before solving; negative basis integrals are retained. The
source is the unnormalized Hann-windowed sine in `fem.tone_burst`. Raw unit
drive traces are multiplied by the scenario's force and TX/RX coupling gains;
current-only channel drift applies only to `current`. Stochastic noise and
nonfree supports are rejected for this numerical reference.

The worker solves `M*a + K*u = f` with beta=1/4 and gamma=1/2. It factors
`M + dt^2*K/4` once, keeps the consistent mass and free rigid-body modes,
checks the relative linear residual at every step, and records
`(v.T*M*v + u.T*K*u)/2` at every internal time. Saved waveforms use exactly
the declared sample grid, with no amplitude fitting, phase shift or
interpolation. Symmetry and positive mass diagonal are checked on import;
positive-definite mass and positive-semidefinite stiffness rely on validated
physical assembly, rather than being inferred from that diagonal check.

## Independent checks and the native cache limitation

A coupled two-coordinate consistent-mass system with nonzero initial state
and time-dependent forcing agrees with an independent adaptive DOP853 solve
at second order. An unforced system containing elastic and rigid motion
conserves its discrete physical energy. Sparse import tests check native
permutation recovery and reject duplicate entries, invalid dimensions and
malformed CSR archives.

On a 75-node, 24-tetrahedron P2 block, independently exported Elmer operators
and this integrator reproduce an actual native Elmer Hann/aperture trace
to **1.74e-12 relative L2** at dt=125 ns. The maximum relative linear residual
is 1.22e-15; postburst relative energy range is 9.66e-14. Matching mesh file
hashes establish that both paths use the same discrete system. Regression
also repeats an independent native P2 aperture comparison from scratch.
Agreement verifies the integration/interface on that system; it is not
continuum or finite-plate spatial validation.

The tested portable Windows release crashes on the second time step with
`Constant Bulk System`/`Constant Bulk Matrix` reuse. Full reassembly succeeds;
isolated factorization-only reuse and sparse SaveScalars output also succeed.
The production waveform path therefore exports operators once and avoids
native bulk reuse. Failed inputs, logs and isolation results are retained.

In the pinned release's
[StressSolve.F90 AddGlobalTime](https://github.com/ElmerCSC/elmerfem/blob/5a8de867068be0568f09af40fb90fee300dfbede/fem/src/modules/StressSolve.F90#L1620),
`A2` is allocated with saved work arrays but omitted from their `SAVE` list.
Its use on a later call is a plausible explanation for the observed crash.
This is a source-level inference, not a patched-binary confirmation. No
vendor binary or global installation was changed.

Native transient controls can separately request factorization-only reuse
using explicit `Logical False` settings. SaveScalars subsampling uses
`Exec Intervals Offset = stride - 1`, as required by the timestep gate in
[MainUtils.F90](https://github.com/ElmerCSC/elmerfem/blob/5a8de867068be0568f09af40fb90fee300dfbede/fem/src/MainUtils.F90).
Actual output times and traces are checked against full output.

## Integrity and resource bounds

Each plan binds the scenario, settings, actions, Python/NumPy/SciPy versions,
all local solver source hashes and native executable/core/stress-library
hashes. A deterministic `source-snapshot.zip` preserves the actual Python
source used. Worker jobs bind every native operator/load file, validate
paths and source hashes, and write fingerprinted reports and trace hashes.
Imports have bounded counts/file sizes and never unpickle arrays or expand
the sparse operators into dense matrices.

The checked float64 group cache keys mesh/load/source/kernel/runtime/time
recipes. Only completed, checksum-verified groups are reusable. Cache recovery
still rebuilds and checks the mesh/apertures; it records zero new solves.
Unfinished attempts need a new output directory. A parent process monitors
the worker and all observed descendants and can terminate native allocation
or factorization when time/RSS/system limits are reached. GPU readings are
system telemetry; this solver uses CPU. CPU temperature is unavailable here.

## Measured finite-plate pilot, 2026-11-04

The final-source run used the same 40 x 40 x 3 mm intact control as the
[earlier P1 refinement](waveform-validation.md): eight 3 mm radius apertures,
one transmitter, seven receivers, a 100 kHz two-cycle pulse, and 128 samples
spaced 0.5 microseconds apart. The internal step was 125 ns (508 steps).

| Quantity | Measured value |
| --- | --- |
| Target mesh size / P2 nodes / tetrahedra | 3 mm / 3,777 / 1,928 |
| Degrees of freedom | 11,331 |
| Stiffness / mass nonzeros | 785,251 / 261,837 |
| LU factor nonzeros | 3,932,800 |
| Maximum relative linear residual | 5.98e-16 |
| Postburst relative energy range | 9.46e-15 |
| Total wall time | 9.69 s |
| Sampled monitor/process-tree RSS | 302.2 MiB |
| System RAM / GPU memory / GPU temperature | 63.7% / 921 of 8,192 MiB / 47 C |

All **177 regression tests passed** with optional real Elmer checks enabled
in 104.72 s. Sampled combined RSS peaked at 783.5 MiB and system RAM at 66.2%;
GPU readings stayed at 921 MiB and 47 C. No training ran. Compact hashes,
inputs, resources and results are in
[evidence/elmer-waveform-debug.json](evidence/elmer-waveform-debug.json).
The seven raw receiver traces and physical energy curve were exported as
PNG/SVG and visually checked for readable axes, units and provisional labels.

These measurements establish a working bounded reference path. Separate
spatial/time refinement, defect scattering, bonded PZT/adhesive/electrical
domains and measured intact/defect validation remain required. The 500 mm
research plate has not been launched through this debug configuration.
