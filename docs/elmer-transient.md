# Independent transient interfaces and time refinement

Elmer's 3D transient solver now has bounded checks for force units, named
scalar probes, equivalent nodal loads and second-order time integration.
Both linear and quadratic tetrahedra passed. This establishes a tested
interface for future mechanical references; it does not resolve finite-plate
spatial accuracy or supply a PZT voltage model.

```powershell
.\.venv\Scripts\python.exe -m traxis verify-elmer-transient --elmer-home outputs/reference-tools/elmer-26.1/ElmerFEM-nogui-nompi-Windows-AMD64 --output outputs/elmer-transient-reference
```

Use the existing solver described in [elmer-reference.md](elmer-reference.md)
and a fresh output directory. The command freezes its plan before assembly,
then runs 18 short transients, two operator exports and one quadratic mesh
conversion. Defaults are 60 seconds and 512 MiB sampled RSS for the monitor
plus external process tree, with system RAM and best-effort GPU/temperature
limits. It launches no training or dataset generation. Completed failed
screens return 2; operational failures return 1 and retain the partial report.

## Force and probe definitions

The free benchmark block is 40 x 30 x 3 mm. A 2 x 2 x 1 structured division
gives 24 tetrahedra, with 18 P1 nodes or 75 P2 nodes after ElmerGrid conversion.
There are no supports, damping, transducers or displacement constraints.

The first load is a uniform z-directed volume force
`b(t) = 1e6 * sin(2*pi*100000*t)` in N/m³. Its rigid-body acceleration is
`b/rho`, with rho=2700 kg/m³. A spatially uniform acceleration translates a
free body without strain; this check therefore tests mass, units and time
integration rather than elastic-wave propagation. The analytic displacement
from zero displacement/velocity is `a * (t/Omega - sin(Omega*t)/Omega²)`.
The sine starts with zero force, consistent with zero initial acceleration.

The second load is the equivalent discrete nodal force
`M * ez * (1e6/rho) * sin(Omega*t)` in N. Elmer receives it through targeted
`Displacement i Load` conditions. This exercises the force interface needed
for later finite mechanical apertures without approximating a PZT here.
P2 has 18 negative equivalent nodal coefficients in this case. Such signs
arise from quadratic basis integrals even for a positive uniform volume load;
they are preserved. Clipping them would change the load. The consistent mass
remains positive definite.

The third load excites the first nonrigid generalized eigenmode of the fixed
Elmer-assembled system. Before any transient, its vector is normalized so
its largest component at the chosen probe is one. The force is
`M * phi * 1000 * sin(Omega*t)` N, with `Omega = 0.7*omega0`.
The scalar reference solves `q'' + omega0²*q = 1000*sin(Omega*t)` from rest.
This analytic ODE tests temporal error for a discrete elastic mode. It is
not an independent continuum eigenfrequency or spatial-convergence result.

The solver explicitly uses Bossak alpha=0, giving average-acceleration
Newmark with gamma=1/2 and beta=1/4. It uses direct UMFPACK solves and keeps
the mesh fixed. Elmer's default Bossak alpha differs, so that default is not
silently used. Relevant implementations are
[SolverUtils.F90](https://github.com/ElmerCSC/elmerfem/blob/5a8de867068be0568f09af40fb90fee300dfbede/fem/src/SolverUtils.F90),
[TimeIntegrate.F90](https://github.com/ElmerCSC/elmerfem/blob/5a8de867068be0568f09af40fb90fee300dfbede/fem/src/TimeIntegrate.F90)
and [Stress.F90](https://github.com/ElmerCSC/elmerfem/blob/5a8de867068be0568f09af40fb90fee300dfbede/fem/src/Stress.F90).

SaveScalars supplies time, mean/min/max z displacement and the selected node's
component values. The reader maps columns through their metadata names and
checks the requested times, row counts and finite values. It rejects changed
time grids, missing probes and duplicate names. It never infers a column from
its position. Elmer's convenience `Displacement Velocity` field is not used:
the inspected implementation computes a displacement difference, rather than
exporting the integrator's internal velocity directly.

## Refinement and fixed criteria

Each case runs three drive cycles with 32/64/128 steps per cycle. Errors are
evaluated at the same 96 coarse-grid times by exact subsampling of finer
outputs. There is no interpolation, fitted amplitude, phase or time shift.
The full unmodified scalar outputs, mesh/SIF inputs, loads and hashes remain
available alongside the report.

The predeclared finest relative L2 tolerances are 0.1% for rigid translation
and 0.5% for the elastic mode. Both successive observed orders must lie in
[1.7,2.3]. Uniform volume/nodal traces must agree within 1e-8 relative L2;
the rigid body's maximum nodal spread must stay below 1e-9 of reference RMS.
All results retain `ready_for_training_data=false`.

## Measured pilot, 2026-11-04

| Case, either element order | Relative L2 error at 32 steps/cycle | At 64 | At 128 | Observed time order |
| --- | --- | --- | --- | --- |
| Uniform volume force | 0.3260% | 0.08147% | 0.02036% | 2.0002–2.0006 |
| Equivalent nodal force | 0.3260% | 0.08147% | 0.02036% | 2.0002–2.0006 |
| Discrete elastic mode | 6.027% | 1.510% | 0.3777% | 1.9969–1.9993 |

The maximum equivalent-load discrepancy was 1.04e-12 relative L2. Both
element orders passed all criteria. The coarse mesh's first elastic frequency
was 48,265 Hz for P1 and 8,219 Hz for P2. Their large difference is a reminder
that the time checks do not certify either value as a physical plate mode.

The complete pilot took 31.24 seconds with sampled combined RSS 124.3 MiB,
system RAM 61.8%, GPU memory 921/8,192 MiB and GPU temperature 47 C. All
computation used CPU; GPU readings describe the system, not allocations by
Elmer. CPU temperature was unavailable. Compact evidence is in
[evidence/elmer-transient-debug.json](evidence/elmer-transient-debug.json).

Analytic references are also checked against an independent adaptive DOP853
ODE solution, including exact/near resonance and the small-time jerk limit.
The optional real-solver regression repeats both element orders and checks
the force equivalence, temporal orders and preservation of negative P2 loads.
Enable it with the process-local `TRAXIS_ELMER_HOME` environment variable.

All 161 regression tests passed with the real Elmer checks enabled. The full
monitored suite took 90.95 seconds, peaking at 781.3 MiB sampled combined
RSS and 64.8% system RAM. GPU memory was 921/8,192 MiB with a maximum observed
temperature of 47 C; CPU temperature was unavailable.

Next requirements remain finite-area P2 source/probe integration, converged
finite-boundary/recess waveforms, appropriate bandwidth, supports and actual
PZT/adhesive/electrical inputs. This command supplies no research waveform
bank and performs no calibration on physical holdouts.
