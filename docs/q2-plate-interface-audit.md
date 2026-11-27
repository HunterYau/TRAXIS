# Eight-ring Q2 PZT / aluminum interface investigation

Branch: research/q2-plate-interface-audit. No production physics, solver or LTS code changed. No full plate simulations. All small solves use FP64.

## Findings

1. Displacement interpolation reproduces constant, linear and quadratic fields to roundoff. Transpose force transfer preserves resultant force, first moments and mechanical work. These tests do not indicate a sign, scaling or energy-conservation bug.
2. The production fifth-order aluminum mesh has 4 mm elements, not a low-order 4 mm nodal spacing. Its analytical A0 input is imperfectly interpolated. Refining host sampling improves the isolated receiver voltages while keeping the eight-ring PZT unchanged.
3. The perfect tie uses nodal elimination and a transpose variational force map, not a penalty stiffness. There is no independent interface traction quadrature or duplicate force contribution. Equal displacements are imposed at the slave nodes. Between those nodes, the Q2 PZT trace and fifth-order host trace are not identical. This is a real discretization mismatch, though the sampled gaps are small.
4. Strong tie restricts the PZT contact to the host interpolation space. That can stiffen unresolved interface deformations, but a roundoff patch test or exact work transfer alone cannot prove absence of locking. These tests do not establish excessive artificial stiffness in production.
5. The small bonded patch has strongly mesh-dependent resonances and voltages. It demonstrates sensitivity to host resolution and loading, but its finite reflecting boundaries and second-order host are not the production plate. Its large differences cannot be assigned directly to the production interface.
6. Elementwise generated-charge contributions strongly cancel for A0 near 150-200 kHz. This amplifies relative voltage sensitivity. It is not proof that all previously observed mesh differences are physical.

## Production interpolation and isolated receiver

Same NCE51 profile, 10 mm diameter / 0.5 mm thickness disk, fixed Q2 8 rings / 64 polygon sides / 1 thickness cell, Gauss order 6 and full consistent mass. Host: E=71 GPa, nu=0.33, density=2700 kg/m3, 1 mm thickness. Analytical Lamb displacement amplitude scale is 1 nm. Both actual centers were checked.

| Frequency kHz | S0 voltage difference % | A0 voltage difference % | Max A0 absolute difference mV |
|---:|---:|---:|---:|
| 50 | 0.00000000 | 0.001192 | 0.00070755 |
| 100 | 0.00000016 | 0.008090 | 0.00414742 |
| 150 | 0.00000167 | 0.129482 | 0.01815703 |
| 200 | 0.00000833 | 0.214159 | 0.05816341 |
| 250 | 0.00002685 | 0.198313 | 0.11439536 |
| 300 | 0.00006255 | 0.112438 | 0.07987424 |

The refined 500/1000-cell host maps are coordinate/interpolation metadata only: no global plate matrices, vectors or simulation were built. Each voltage comparison uses direct surface displacement versus interpolation through the production shape functions.

| Host element width mm | Max A0 receiver difference % | Max sampled work error |
|---:|---:|---:|
| 4 | 0.21415852 | 1.78e-14 |
| 2 | 0.00257594 | 4.39e-15 |
| 1 | 0.00001903 | 6.3e-16 |

At 300 kHz, the area-weighted A0 displacement trace gap between slave nodes is 0.05572% (4 mm host), 0.04363% (2 mm host), 0.04329% (1 mm host). The nonzero refined limit includes the unchanged Q2 face approximation. This trace check uses surface quadrature; force work conservation follows the discrete transpose map.

## Bonded patch refinement and cancellation

A 12x12x1 mm aluminum patch is refined uniformly under and around the unchanged disk: 4x4, 6x6 and 8x8 Q2 host cells. GLL host mass matches the production mass treatment; the PZT keeps full consistent mass. Perimeter displacements follow the same analytical S0/A0 fields, with other exterior faces free. This is a bounded surrogate study, not true local refinement within the production plate. There is no damping, so nearby standing-wave resonances can exaggerate changes.

| kHz | S0 4→6 % | S0 6→8 % | A0 4→6 % | A0 6→8 % | A0 cancellation at 8 cells % |
|---:|---:|---:|---:|---:|---:|
| 50 | 8874.168 | 45.803 | 0.712 | 1.095 | 62.529 |
| 100 | 2.708 | 1.575 | 1.444 | 0.068 | 82.326 |
| 150 | 3.205 | 1.309 | 12.069 | 14.577 | 94.382 |
| 200 | 5.497 | 2.019 | 44.065 | 12.948 | 94.284 |
| 250 | 0.993 | 1.858 | 35.587 | 19.054 | 91.926 |
| 300 | 3.642 | 1.417 | 58.941 | 2.828 | 63.653 |

Absolute voltage differences, signed complex voltages, phase changes, generated charge, element-charge cancellation, and six centroid strain RMS components are recorded in the JSON/CSV. Cancellation is 1-|sum(q_element)|/sum(|q_element|), using the electrostatic electrode weighting field. This is generated strain charge; the net free electrode charge in an ideal open-circuit solution is zero. It is not a claim of measurable nonzero open-electrode free charge.

## Free and bonded resonances / port frequency response

Four eigenpairs nearest 150 kHz were computed with full electromechanical open-electrode condensation and consistent PZT mass. These are not the four lowest modes. Eigenpair residuals are checked; mode shapes were not tracked between models, so adjacent list entries are not necessarily corresponding modes.

| Model | Four resonances nearest 150 kHz (kHz) |
|---|---|
| Free PZT | 147.626, 147.626, 157.755, 157.755 |
| Bonded 4x4 patch | 115.411, 163.906, 176.818, 176.818 |
| Bonded 6x6 patch | 106.978, 151.551, 174.536, 182.123 |
| Bonded 8x8 patch | 147.724, 173.326, 176.704, 181.729 |

A separate identical 1-pC electrical port excitation measures dynamic capacitance and mechanical displacement per volt for free and bonded models at all six frequencies. Values are saved as electrical_port_frf. The bonded perimeter is fixed for this port test; it is analytically driven for the S0/A0 tests. Raw displacement norms are not comparable across differing DOF counts without spatial weighting.

## Likely causes and possible follow-up solutions

- Confirmed numerical contribution: host interpolation of shorter-wavelength A0 input. Local host refinement or improved interface approximation could reduce it.
- Confirmed approximation: nonmatching polynomial traces between slave tie nodes. A mortar or Nitsche comparison and a conforming reference patch would test whether a weak tie improves accuracy.
- Confirmed sensitivity: element-charge cancellation and bonded resonances amplify small amplitude/strain changes. Compare absolute voltage/charge as well as relative percentages.
- Unresolved: how much production error comes from artificial interface stiffness versus legitimate PZT mass/stiffness loading. A matched fifth-order local-patch refinement and tracked modes are needed; the low-order finite patch cannot settle this.
- No evidence found in these checks of lost mechanical work, wrong force scaling, failed polynomial interpolation, or an altered LTS recurrence.

No fixes were implemented. Existing fixed LTS time steps were not assessed by these frequency-domain tests.

## Primary references

[MOOSE node/face and mortar interface theory](https://mooseframework.inl.gov/modules/contact/) describes shape-function force transfer and the alternative mortar approach. [COMSOL continuity documentation](https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_solid.07.128.html) describes pointwise, weak and Nitsche ties for nonconforming meshes. These references motivate possible follow-up tests; they do not validate this particular implementation.

All results: outputs/q2-interface-audit-002/. Initial preliminary runs remain preserved in outputs/q2-interface-audit-001/. No RunPod activity occurred.
