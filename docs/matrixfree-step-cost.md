# First full-size acceleration solve

The matrix-free operator fits the guarded full-size allocation path, but a
time-domain solve needs many applications. This command measures the actual
first average-acceleration Newmark equation, using the retained intact mesh
and a checked circular mechanical load. It does not shorten a research scan
and call the result an accepted waveform.

```powershell
.\.venv\Scripts\python.exe -m examples.matrixfree_step --capacity outputs/matrixfree-fullplate-capacity-20261107 --settings configs/matrixfree-first-step.json --output outputs/matrixfree-first-step --max-seconds 180 --max-rss-mib 1024
```

Use a fresh output directory and existing completed capacity evidence. The
command checks the original plan/report/mesh hashes and geometry implementation,
then rebuilds current operators. Old sparse matrices or unchecked native
buffers are not reused. The new plan binds all source, both example drivers,
native runtimes, input hashes and the source snapshot.

## Equation and mechanical source

With zero initial displacement/velocity and a Hann source that is zero at
time zero, initial acceleration is zero. The first step is exactly

`(M + dt^2 K/4) a_1 = f(dt)`,

followed by `u_1 = dt^2 a_1/4`, `v_1 = dt a_1/2`. K is the full vector
elastic operator and M is consistent mass. The same checked CG implementation
now serves this probe and the existing Newmark integrator. Its strict solver
status and independently recomputed final residual remain mandatory.

The full-size settings use the retained 500 x 500 x 3 mm P3 mesh, a 3 mm-radius
normal-force disk centered at (20,20) mm, 100 kHz/two-cycle Hann drive with a
nominal 1 N scale, and dt=125 ns. Actual first-step force is 3.02446e-5 N.
The load is mechanically work-conjugate to averaged displacement; it is not
PZT actuation or electrode voltage. Its area and every represented centered
monomial are checked before solving.

Larger footprints now require explicit scalar/element/workspace limits.
Existing small defaults remain, and an extraction-memory screen runs before
materializing the native volume/top-face arrays. The full-size request uses
the declared matrix-free count limits and a 256 MiB extraction screen, with
the unchanged outer memory guard.

`progress.json` saves every completed CG iteration's count, elapsed time,
iterate norm and operator-application counts. These norms are **not residual
estimates**. Only a returned solve has an independently checked true residual.
An interrupted solve must not be described as converged. No partial iterate
is exported as an accepted state or used for training.

After convergence, separate K/M actions check discrete energy against the
trapezoidal external work, `f(dt).u_1/2`. A successful probe saves acceleration
and the mechanical load. Even success would cover only one numerical step,
not a finite-plate arrival, mesh/time convergence or physical agreement.

## Measured result, November 7

The full-size run reached the 180-second wall guard before completing its
first acceleration equation. It retained fourteen completed CG iterations,
the last at 170.47 worker seconds. Operators/diagonals were ready at 25.91
seconds and the integrated/moment-checked source at 39.24 seconds. Completed
iterations cost roughly 9-10 seconds each. No final true residual is available,
so this is a resource/cost stop, not a completed numerical-accuracy screen.

Peak sampled combined RSS was 494.48 MiB, system RAM 45.4%, available memory
at least 8,860 MiB, GPU system usage 894/8,192 MiB and GPU temperature 57 C.
All calculation used CPU; CPU temperature was unavailable. The guard retained
the plan, source snapshot, load, stage/iteration progress and monitor result in
`outputs/matrixfree-fullplate-first-step-20261107`. Zero steps completed and
`ready_for_training_data` remains false. Raising the wall limit until this
passes would not establish an affordable dataset-generation workflow.

This confirms a substantial transient-cost problem for this single-threaded,
diagonally preconditioned implementation despite its modest memory use.
Efficient preconditioning/operator execution or a different independently
verified elastic solver remains necessary before research-bank generation.
Full-size wave accuracy, defect response and calibrated PZT voltage are open.

## Checks

Twelve focused checks pass, including independent coupled-ODE/Newmark tests,
strict/false-success CG rejection, initial-guess preservation, progress on
nonconvergence, explicit footprint limits, and the actual native first-step
comparison to sparse acceleration and direct Newmark on a small P2 mesh.
The sparse test is explicitly capped at 2,000 coefficients. An initial debug
fixture had 2,958 coefficients and was rejected; its retained run remains in
`outputs/matrixfree-first-step-native-debug-20261107`. The cap now fails before
solving. The small verification fixture uses an 8 mm target mesh to stay within
that existing bound; the full-size measurement retains its original 5 mm mesh.

All 243 regression tests pass without skips with real Elmer/Gmsh/NGSolve,
including the all-column/direct-transient comparison and zero-new-case cache
recovery. The suite took 442.15 test seconds / 445.27 monitored seconds at
775.54 MiB sampled combined RSS, 46.0% system RAM and at least 8,767 MiB
available memory. System GPU observations peaked at 742/8,192 MiB and 43 C;
calculation used CPU and CPU temperature was unavailable. No full training ran.

The retained native P2 direct-reference mesh has 1,671 vector coefficients.
Its first equation converged in 108 iterations / 0.820 solve seconds, with
true relative residual 7.98e-12, acceleration difference 3.04e-11, probe
difference 1.55e-15 and relative work/energy error 1.11e-15. These are small
discrete checks; its iteration count must not be extrapolated to the full plate.
The actual test's reports and arrays were retained before cleanup, without
another solve, under `outputs/matrixfree-step-direct-reference-20261107`.

The compact [evidence record](evidence/matrixfree-step-debug.json) binds source,
drivers, tests, retained inputs/arrays, the initial direct-cap rejection,
full-size stage/iteration progress and all monitor summaries. The full-size
stop has no final independent residual or accepted state; partial iterate
norms do not establish whether the CG target was reached before interruption.
