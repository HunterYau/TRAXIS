# Coupled piezoelectric reference

`examples.piezoelectric_reference` verifies the electrical/mechanical coupling
needed before a force/displacement plate control can become a PZT experiment.
It is a small independent reference, not a bonded-transducer plate solver or
an identified model of a purchased disc. Its synthetic material coefficients
must not be used as calibrated PZT data.

The constitutive starting point is the stress-charge formulation described in
the [COMSOL piezoelectricity theory documentation](https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_theory.06.043.html).
The [terminal documentation](https://doc.comsol.com/6.4/doc/com.comsol.help.acdc/acdc_ug_electric_fields.07.020.html)
distinguishes imposed voltage, imposed charge and circuit connections. Those
sources support the equations and boundary-condition interpretation; the
numerical comparisons here use NGSolve, not a COMSOL execution.

## Conventions and independent assembly

Engineering strain is ordered `xx, yy, zz, yz, xz, xy`, with doubled shear
strains. The electric field is `E = -grad(phi)`. With a 3-by-6 coupling matrix,

```text
stress = C_E strain - e.T E
D      = e strain + epsilon_S E
```

Both `C_E` and the clamped permittivity must be symmetric positive definite;
all coefficients are real and finite. Density must be positive. The example
contains a synthetic anisotropic material in SI units and labels it explicitly.

An independent affine-P1 tetrahedron calculation constructs the strain matrix
from the inverse coordinate matrix. Its exact element blocks are

```text
Kuu  = volume B.T C_E B
Buφ  = volume B.T e.T grad(N)
Kee  = volume grad(N).T epsilon_S grad(N)
M    = consistent linear tetrahedron mass, component by component
```

The symmetric electric-enthalpy system has blocks `[Kuu, Buφ; Buφ.T, -Kee]`.
Every block is compared separately, so the much larger elastic entries cannot
hide an electrical error in a norm of the entire mixed matrix. The independent
consistent mass is not the enriched positive nodal mass used in production
plate controls.

An independently assembled native space has three scalar displacement fields
and one potential field. The complete P1 nodal map is verified at every local
vertex before comparing the operators. Native quadrature includes an explicit
integration-order allowance. Affine strain/electric-field energies, six rigid
motions and the constant-potential gauge are also checked.

The two meshes are 40-by-30-by-3 mm blocks with 2-by-2-by-2 cells: 27 vertices,
48 tetrahedra and 108 coupled unknowns each. One is rectangular and one is
affinely skewed. These dimensions are verification fixtures, not a replacement
for the research plate or the proposed disc geometry.

## Electrode condensation and reciprocity

The bottom electrode is grounded, the top is equipotential, and the remaining
electrical boundaries have zero free charge. Interior potentials are eliminated
without discarding their piezoelectric response. All bottom mechanical
displacements are clamped for this reference. Our algebra gives the port

```text
M u'' + Ksc u + g V = F
Q = Cp V - g.T u
```

`Cp` is positive. The rectangular zero-displacement capacitance agrees with
`epsilon_zz * area / thickness` (4 nF for this synthetic fixture). Prescribed
voltage actuation is compared with a separately solved native mixed system.
For an uncharged isolated top electrode, `V = g.T u / Cp`, giving
`Koc = Ksc + g g.T / Cp`. The open-circuit displacement and generated voltage
are compared with a native mixed solve that imposes the electrode constraint
directly, rather than reusing the condensed solve as its own reference.

Mechanical and electrical signs are reciprocal: reversing poling changes the
voltage/coupling sign while preserving passive stiffness and capacitance.
Zero coupling eliminates the port force. Tiny negative eigenvalues of the
numerically subtracted rank-one stiffness increment are retained and assessed
relative to the stiffness scale; they are not clipped or called physical
negative stiffness.

## Passive and driven circuit checks

With nonnegative conductance `G` and current injected into the top terminal,

```text
Cp V' - g.T u' + G V = I
energy' = F.T u' + V I - G V^2
```

This energy relation follows from the two port equations above. It is checked
for an isolated charged port, a resistor-loaded port and a current-driven
resistor-loaded port. A complete mass-normalized modal basis is used on each
small mesh; no mechanical modes are truncated. The energy-scaled state has a
skew-symmetric lossless generator and a negative resistive contribution.

Implicit midpoint propagation at 32, 64 and 128 steps is compared with a
separate SciPy matrix exponential of the retained linear system. Energy
balance includes work supplied by the current source and resistive loss.
The isolated case also checks conserved electrode charge. The intervals are
about 0.35 microseconds, chosen from the largest discrete modal frequency to
identify the integrator's rate. They are not a 500-microsecond plate experiment
or evidence about a real transducer's operating band.

## Measured result and reproducibility

The complete reference passes on both meshes. Maximum relative block
discrepancy against native assembly is 2.51e-14. Maximum affine-energy error is
2.69e-14. Open-circuit native/condensed displacement and voltage discrepancies
are below 2.19e-14. The six three-level circuit cases have error ratios between
3.99557 and 3.99896, inside the declared 3.8-to-4.2 second-order screen. The
largest finest endpoint error is 5.27e-5 relative. Retained energy and isolated
charge balances pass their 1e-10 screens.

Three focused CPU tests pass: polarity/zero-coupling and analytic capacitance;
isolated/passive/driven circuit power; and invalid material/electrode/load
rejection. This is a separate group, not a new complete-suite count. The last
complete integrated regression remains the earlier 268-test result.

Two initial reporting attempts failed because NumPy boolean acceptance values
were not JSON serializable. The first parent's missing-result error concealed
the worker traceback; the parent now retains its worker result and reports a
nonzero exit before reading a missing receipt. Both failed logs remain intact.
Standard Python acceptance flags fix serialization. Subsequent complete runs
pass; the final one additionally saves all modal states/matrices for audit.

```powershell
.\.venv\Scripts\python.exe -m examples.piezoelectric_reference --output outputs/piezoelectric-check
```

The wrapper caps the process tree at 512 MiB and 120 seconds, preserves the
85% system-RAM/2 GiB available-RAM guards, and runs CPU assembly with one
thread. It does not allocate CUDA data. Source/runtime identities, a source
snapshot, raw operators, modal matrices/states, voltage histories and monitor
receipts are retained. GPU observations include the concurrently running
full-plate control and must not be attributed to this CPU reference. CPU
temperature remains unavailable on this PC.

The measured final run is `outputs/piezoelectric-complete-evidence-20261109`.
An additional archive audit reconstructs the condensed port, verifies modal
mass/stiffness identities, rebuilds each circuit generator, recomputes every
matrix-exponential endpoint and checks reported errors against raw arrays.
The compact record is `evidence/piezoelectric-reference-debug.json`, fingerprint
`a73515fffd022fc786ea7880242b8966813d40eb829a2dd9df5cddbd98be5605`.
Its collector is `outputs/piezoelectric-tools-20261109/collect_reference.py`.

## Remaining physical work

This establishes a checked coupling/port reference. It still needs an actual
transducer geometry/material identification, poling and electrode data,
plate/adhesive mechanical coupling, the drive and receiver circuit response,
appropriate damping and space/time convergence. Bonded PZT measurements and
intact-plate physical agreement remain necessary. It accepts no training bank
and performs no reconstruction or policy training. The frozen full-plate
mechanical studies retain their existing sources and controls.
