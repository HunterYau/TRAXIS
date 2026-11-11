# Small nonaffine tensor reference

`examples.mapped_tensor_reference` verifies the operator geometry needed
before extending the Cartesian tensor solver to recesses. It is a bounded
CPU/native reference, not a mapped CUDA solver or a recess mesh. The frozen
intact wave sources are unchanged by this separate example.

## Manufactured geometry

Two connected reference hexahedra span 40-by-30-by-3 mm. With reference
coordinates x,y,z and lengths w,h,t, the physical map is

```text
X = x + 0.1*y*z/t
Y = y + 0.05*x*x/w
Z = z*(1 + 0.12*x/w - 0.08*y/h + 0.04*x*y/(w*h))
```

This is a nonaffine polynomial map, represented independently in native
VectorH1 degree two. NGSolve's deformation API changes the actual geometry
used in assembly. Physical nodal positions and continuity of shared scalar
shape evaluations are checked before transforming native operators into
the independent nodal basis.

The independent implementation evaluates the analytic Jacobian at every
GLL point, transforms gradients and assembles local elastic energy and
positive diagonal mass. No single constant-Jacobian shortcut is used.
The minimum sampled determinant is 2.043e-7 and maximum Jacobian condition
number 11.244. The measured coordinate discrepancy is below 7e-18 m.

## Independent results

The following are whole-matrix native comparisons, not a subset of columns.
The declared comparison tolerance is 1e-10.

| Degree | Vector coefficients | Stiffness relative discrepancy | GLL mass relative discrepancy |
| --- | ---: | ---: | ---: |
| Q3 | 336 | 2.60e-15 | 2.40e-15 |
| Q4 | 675 | 3.25e-15 | 4.29e-15 |
| Q5 | 1,188 | 4.55e-15 | 7.99e-15 |

All three have positive mass and exactly six rigid modes. Independent native
seven-lowest-eigenvalue checks give seventh values 2.20e-6, 7.23e-7 and
3.00e-7 of the generalized Gershgorin upper bound, while the first six are
within the declared 1e-12 normalized zero tolerance. The maximum native
rigid-field residual is 6.72e-17 relative.

A separate auditor integrates the closed-form dimensionless Jacobian
determinant exactly. Its volume ratio is 20491/20000, giving 3.68838e-6 m^3.
All mass-derived volumes agree within 8.89e-16 relative. A manufactured
affine physical displacement has exact constant strain; native energies
agree with the analytic volume/constitutive expression within 1.27e-14.
The auditor recomputes all matrix differences and native spectral checks
from retained arrays without calling the producer's assembly routine.

The native worker completes in 3.561 monitored seconds, peaking at
353.90 MiB process-tree RSS. System RAM peaks at 56.6%, with 7,037.89 MiB
minimum available. The 1,881 MiB / 77 C GPU observations include the
independent concurrent Q5 plate wave; this reference is CPU-only.
CPU temperature is unavailable.

## Retained failures and evidence

The first attempt used a layout-only Jacobian property on the bounded grid
and failed with AttributeError. The corrected affine map is computed from
the declared grid lengths, cell counts and transform. The second attempt
used a compound scalar space for native geometry deformation; the installed
API rejected it because it needs an actual vector space. VectorH1 supplies
the required dimension. Both failures and source snapshots remain under
`outputs/mapped-tensor-native-20261110` and its `-v2` directory.

The `-v3` run passes operator/energy checks. The final `-v4` run adds the
explicit seven-eigenvalue nullity check and retains all raw operators,
mass, physical nodes, rigid fields and manufactured displacement. Compact
evidence is `evidence/mapped-tensor-reference-debug.json`, fingerprint
`eb6945c4204dd68e4f3881629c4db80d06f1c853e92be27e2338424e78f9eb9b`.
The independent auditor is `outputs/tensor-wave-tools-20261109/audit_mapped.py`.

```powershell
.\.venv\Scripts\python.exe -m examples.mapped_tensor_reference --output outputs/mapped-reference
```

This verifies one manufactured geometric map through three solution degrees.
Actual conforming circular-recess topology, geometry refinement, mapped GPU
storage/action, physical port integration on nonrectangular traces,
stability and complete wave convergence remain implementation work. See
tensor-recess-design.md. No waveform bank or training acceptance follows
from these local operators.
