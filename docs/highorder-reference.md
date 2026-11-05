# Independent high-order mechanical assembly

The optional NGSolve reference now has bounded P1-P4 assembly checks on
rectangular and affine-skewed blocks. It independently matches corrected
Elmer P2 operators and exact polynomial energies. A separate
[circular aperture check](highorder-apertures.md) now verifies P1-P4 loads,
receivers and coarse pulse traces. It does not generate PZT voltages. These
operator checks do not establish spatial convergence or physical acceptance.

NGSolve supplies high-order H1 spaces and a Python variational interface,
with an official [3D elasticity example](https://docu.ngsolve.org/v6.2.2606/i-tutorials/wta/elasticity3D.html).
Its [installation instructions](https://ngsolve.org/installation.html) support
pip installations. This project tested the official Windows x64 / CPython
3.12 wheels for release 6.2.2606; native work runs in an isolated process.

## Reproduce the check

Install the core dependencies first. The optional portable requirement is
`requirements-highorder.txt`. For the exact Windows environment used here:

```powershell
.\.venv\Scripts\python.exe -m pip --isolated --disable-pip-version-check install --no-deps --only-binary=:all: --require-hashes --no-cache-dir --index-url https://pypi.org/simple -r requirements-highorder-win-lock.txt
.\.venv\Scripts\python.exe -m traxis verify-highorder-reference --elmer-home outputs/reference-tools/elmer-26.1/ElmerFEM-nogui-nompi-Windows-AMD64 --output outputs/highorder-reference
```

Use a fresh output directory and the project-local Elmer setup described in
[elmer-reference.md](elmer-reference.md). The four pinned wheels are
NGSolve/Netgen 6.2.2606, netgen-occt 7.8.1 and ngsolve-openblas 0.3.33.
The installed NumPy 2.5.4 and SciPy 1.18.1 were preserved. Installation
temporary files stayed inside the project; no global PATH or registry
configuration was changed. `pip check` reports no broken requirements.

The command defaults to 60 seconds and 512 MiB sampled monitor/process-tree
RSS, with system memory and best-effort GPU/temperature guards. Completed
failed numerical screens return 2; operational errors return 1 and retain
partial reports. All results set `ready_for_training_data=false`.

## Formulation and verification

Both blocks use the same 2 x 2 x 1 division, giving 18 vertices and 24
tetrahedra. One is a 40 x 30 x 3 mm box; the other is a fixed affine skew
with independently changed elastic constants/density. The skew is an
assembly benchmark, not a simulated research plate. All boundaries are
natural/free, with no displacement constraints, damping or electrodes.

The assembled forms are `integral(2*mu*eps(u):eps(v) + lambda*div(u)*div(v))`
and `integral(rho*u.v)`. They use three scalar H1 component spaces and retain
all degrees of freedom without static condensation. Full CSR matrices are
copied from the native library, checked for symmetry/finite values, and
saved without pickle or dense expansion. Topological counts and conservative
assembly estimates are checked before native allocation.

Coefficients are component-major and hierarchical above degree one.
They must not be passed to the existing node-major P2 aperture interfaces.
For independent comparison, a checked square basis evaluation maps H1
coefficients to the same physical P1/P2 nodal displacements. Mass and
stiffness are compared under that basis transformation.

The fixed criteria require relative operator/energy errors below 1e-10,
six rigid modes, positive-definite mass in the generalized eigensolve, and
nonincreasing first six elastic Ritz frequencies as degree increases
(1e-9 relative roundoff allowance). Exact affine-box integrals check nine
degree-matched displacement monomials, mixed dilation and all six rigid
fields at each degree. These checks exercise shear, volume, component
coupling, geometry mapping and the degree-four/eight mass products.

The P2 mass also has a separate exact barycentric-product reference. This
comparison exposed and corrected the earlier Elmer quadrature configuration;
see [elmer-mass-quadrature.md](elmer-mass-quadrature.md). The original failed
screen and its inputs are preserved alongside the successful corrected run.

## Measured assembly pilot, 2026-11-04

All eight geometry/degree combinations passed. The largest polynomial mass
and stiffness errors were below 4.25e-14 and 4.97e-14. Maximum P1 operator errors
were 8.49e-16 for stiffness and 2.06e-14 for mass. P2 agreement with corrected
Elmer was below 2.63e-14 for both operators. Against the exact nodal P2 mass,
Elmer and NGSolve errors were at most 2.86e-15 and 2.55e-14, respectively.

| Degree | Scalar / vector degrees of freedom | Rectangular first elastic frequency |
| --- | --- | --- |
| P1 | 18 / 54 | 48,265.3 Hz |
| P2 | 75 / 225 | 8,209.19 Hz |
| P3 | 196 / 588 | 7,875.57 Hz |
| P4 | 405 / 1,215 | 7,767.82 Hz |

These are discrete free-block frequencies. The degree sequence demonstrates
the variational refinement check, not an accepted continuum value or a
verified 100 kHz finite-plate waveform. The later circular aperture and
discrete transient checks are described in highorder-apertures.md;
mesh/time convergence and physical PZT calibration remain required.

The complete native pilot took 3.72 s with sampled combined RSS 291.3 MiB
and system RAM 70.4%. System GPU observations were 1,386/8,192 MiB and
45 C; computation used CPU. CPU temperature was unavailable. The plan binds
the full Python source snapshot, package versions and installed native
library hashes, Elmer executables/stress library/element definitions, all
mesh/operator inputs, basis transforms and saved operator archives.

Compact evidence, including installation provenance and the original failed
mass comparison, is in
[evidence/highorder-reference-debug.json](evidence/highorder-reference-debug.json).

All 197 regression tests passed with both native reference libraries enabled
in 134.49 s. Sampled combined RSS peaked at 771.2 MiB and system RAM at
72.6%; GPU readings were 1,386/8,192 MiB and 45 C. CPU temperature was
unavailable. The first full run stopped one native integration test at its
512 MiB allowance after counting memory retained by the test parent. That
test now uses the suite's 1 GiB combined allowance; the standalone command
keeps its verified 512 MiB default. Both receipts are retained.

A corrected one-transmitter P2 waveform control also completed through the
production aperture/export/integration path: 3,777 P2 nodes, seven receivers,
508 internal steps at 125 ns, 8.37 s and 256.5 MiB sampled combined RSS.
Its maximum linear residual was 7.62e-16 and postburst relative energy range
1.15e-14. Its cache identity includes the element-definition hash. This is
a working corrected trace path, with no spatial/physical acceptance claim
and `ready_for_training_data=false`. No full training ran.
