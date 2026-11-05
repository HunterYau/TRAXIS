# Circular mechanical footprints in higher-order spaces

The optional NGSolve path now integrates circular loads and displacement
receivers in its native P1-P4 coefficient basis. All 48 footprint cases
passed the fixed checks. P2 pulse traces agree with fresh, corrected native
Elmer transients to within 1.7e-12 relative error on the same coarse mesh.
These are mechanical interface checks. Spatial/time convergence, bonded-PZT
electromechanics and physical calibration remain required before research
training data can be accepted.

## Reproduce

Use the project-local optional setup in
[highorder-reference.md](highorder-reference.md) and an existing portable
Elmer installation. No additional packages are needed for this check.

```powershell
.\.venv\Scripts\python.exe -m traxis verify-highorder-patches --elmer-home outputs/reference-tools/elmer-26.1/ElmerFEM-nogui-nompi-Windows-AMD64 --output outputs/highorder-apertures
```

Choose a new output directory. The defaults are 60 seconds and 512 MiB
sampled monitor/process-tree RSS, with system memory and best-effort
temperature/GPU guards. Numerical failure returns exit code 2; an operational
error returns 1 and retains partial reports. All results explicitly retain
`ready_for_training_data=false`. No training updates are performed.

## Integration and basis

The load is a spatially uniform, unit-total-force traction on a circular
part of the flat top surface. Its work-conjugate receiver measures the
same area-averaged displacement component. For a footprint of actual
covered area A and basis function phi_i, both use
`w_i = integral(footprint, phi_i)/A`. The pulse is a 1 N, two-cycle Hann
burst at 100 kHz. This does not represent a specified PZT voltage drive.

The native coefficients are component-major and hierarchical. Their
weights can be negative; summing every coefficient is not a partition of
unity check. Reproducing constant displacement fields checks the resultant
force correctly. P2 covectors are compared with the existing analytic
nodal interface using `w_native = T.T @ w_nodal`, where T maps native
coefficients to physical nodal displacements.

Green's theorem integrates disk-centered monomials through degree four.
For `x^a*y^b`, the radial boundary flux is divided by `a+b+2`. Triangle
edges contribute polynomial line integrals; circular arcs contribute a
finite Fourier integral. The circle is not polygonized. An affine change
of variables gives face-barycentric moments. Each native shape function's
restriction to the face is interpolated as a degree-matched polynomial,
checked at three additional interior points, and integrated against those
moments. Native vertex ordering and all shape signs are checked explicitly.

The native element/degree-of-freedom access follows NGSolve's
[mesh topology interface](https://docu.ngsolve.org/v6.2.2305/i-tutorials/unit-1.8-meshtopology/meshtopology.html).
The installed 6.2.2606 bindings were inspected and exercised directly.
Only straight tetrahedra and a free, flat, axis-aligned top face are
supported here. Edge-clipped footprints use their actual intersection
area; they do not silently receive the full-disk normalization.

## Checks and scope

The block is 40 x 30 x 3 mm. Three mesh arrangements exercise the same
geometry: 2 x 2 x 1 cells, an independently renumbered/reordered version,
and 3 x 3 x 2 cells. Degrees one through four give 12 systems, with up to
4,563 vector coefficients. Each system tests two interior disks, a half
disk at a straight edge and a quarter disk at a corner, all of radius
3.2 mm. Clipped cases also exercise an oblique force/probe direction.

Every represented centered monomial is checked against its exact disk,
half-disk or quarter-disk average in all three vector components. Separate
pure-Python tests cover full/quarter disk moments, whole-triangle Dirichlet
integrals, all vertex permutations, SI translation/scaling, tangencies and
general cuts against independent adaptive vertical-slice quadrature.

The criteria were fixed before solving: normalized monomial absolute error
below 1e-9, covered area relative error below 1e-10, transformed P2 covector
relative error below 1e-10, and transient/reciprocity errors below 1e-8.
The monomial threshold includes native field projection and cancellation
when evaluating high powers centered on a small footprint.

On the baseline mesh, forward/reverse pulses run at every degree with 128
samples at 0.5 microseconds. P2 is compared to newly run native Elmer
transients using the corrected higher mass integration rule. Elmer's
uniform-volume burst independently checks source timing. Reciprocity,
linear residuals and postburst energy retention are checked as well.
Agreement between discrete solvers does not establish continuum accuracy.

The plan binds Python source hashes and a source snapshot, native package
versions/library hashes, Elmer binaries/element definitions and every
native input. Saved arrays include all footprint covectors, baseline sparse
operators and pulse traces, without pickle. Output hashes are checked by
the parent process; the NGSolve worker is isolated and resource monitored.

The retained pilot and regression measurements are recorded in
[evidence/highorder-aperture-debug.json](evidence/highorder-aperture-debug.json).

The retained pilot completed in 11.42 seconds, with sampled combined RSS
256.4 MiB and system RAM at most 70.8%. The largest normalized monomial
error was 7.28e-11; transformed P2 covectors agreed within 5.74e-16 relative
error. Maximum reciprocity error was 2.47e-13, maximum linear residual
1.17e-14 and postburst relative energy range below 9.82e-14.

All 205 tests in the full regression suite passed with both native solvers
enabled, in 147.96 seconds. After a final CLI exception-handling adjustment,
all five focused tests passed, including a repeated isolated native pilot.
The full suite peaked at 772.9 MiB combined RSS and 73.4% system RAM, with
at least 4,321 MiB available. GPU observations peaked at 1,396/8,192 MiB
and 45 C; the solver used CPU. CPU temperature was unavailable. No full
training cycle ran.
