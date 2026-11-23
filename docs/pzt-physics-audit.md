# PZT physics audit — 2026-11-21

## Conclusion

No defect was found in the inspected linear constitutive equations, tetrahedral
assembly, electrode elimination, ideal-open sensing or reciprocal plate attachment.
That is a scoped source review and numerical verification result, not proof of
complete correctness. The current production path is a coupled 3D small-signal
continuum PZT model. The current experimental reconstruction remains materially
idealized and has not been established as a calibrated real transducer model.

Audited path: `examples/resolved_pzt_queue.py`, default
`configs/pzt/resolved-wandowski-ideal-restored.json`, and the matching run-010
physical settings. Historical planar receiver code is not the active body model.
No solver equations or material settings were changed during this audit.

## Equations and implementation

`traxis/piezoelectric.py` implements the standard linear stress-charge equations:

    T = cE S - e.T E
    D = e S + epsilonS E
    cE = inverse(sE)
    e = d cE
    epsilonS = epsilonT - d cE d.T

These agree with the COMSOL theory reference below. Engineering shear uses
xx,yy,zz,yz,xz,xy ordering. Material matrices are checked for symmetry and positive
definiteness, including clamped permittivity. The NCE51 compliance and d matrices,
relative free permittivities and density were checked against the manufacturer's
Ver1404 complete tables (PDF pages 9–11). Overview and complete tables differ in
some nominal constants. The code consistently derives dependent coefficients
from sE,d,epsilonT rather than combining conflicting printed coefficient sets.

Assembly retains all six strain components, nodal electric potential, anisotropic
elasticity, piezoelectric coupling and a consistent mechanical mass matrix.
Internal electric potentials are eliminated algebraically with equipotential
electrodes. This is electrostatic field condensation, not removal of mechanical
inertia or resonances. It is a standard assumption for this class of model.

The inspected sign convention is internally consistent:

    Q = C V - g.T u
    M uddot + Ksc u + g V = external_force
    Q=0 implies V=g.T u/C and Kopen=Ksc+g g.T/C
    prescribed actuator voltage produces external force -g V

`traxis/pzt_body.py` projects body mass and stiffness into the joined plate/body
coordinates using T.T M T and T.T K T. The receiver also changes plate motion;
it is not a passive point displacement probe. `traxis/gpu_pzt_body.py` retains
consistent body mass in joint patch inverses. The active worker actually calls
these routines and uses g/C voltage probes. The displacement field output is
named `displacement_m` by the generic integrator but these weighted probes are
voltages; the worker explicitly accounts for that convention.

## Realism gaps in the active run

| Feature | Active implementation | Consequence / uncertainty |
| --- | --- | --- |
| PZT shape | 3D polygonal disk, 32 sectors, 6 rings, 3 layers, linear tetrahedra | Numerical approximation; finite resolution |
| PZT mechanics | Full anisotropic, inertial, coupled linear model | Small-signal assumption; no hysteresis, depoling or temperature dependence |
| Material | Nominal NCE51 coefficients | Not measured properties of the individual experimental disks |
| Internal PZT losses | Real, lossless tensors; no active damping | Mechanical, dielectric and coupling losses are omitted |
| Bond | Perfect displacement tie | Real cyanoacrylate compliance, thickness and losses omitted |
| Electrode geometry | Full-face equipotential electrodes | Wrap-around metal, solder and wires omitted |
| Electrical receiver | Ideal isolated zero-free-charge electrode | Actual input impedance, cable response and instrument filtering omitted |
| Actuator | Recorded generator shape normalized to 1 V peak | Actual amplified terminal waveform, source impedance and bandwidth uncalibrated |
| Polarity | Explicit assumed poling sign | Experimental wiring and poling not independently established |
| Plate | Free, lossless, nominal material | Actual supports, damping and alloy uncertainty remain |

Some alternate configurations support parallel resistance/capacitance and a finite
elastic bond. Those features are not enabled in the restored ideal run. The bond
option itself is a coincident thin spring interface with lumped adhesive mass,
not a resolved viscoelastic adhesive volume. The amplifier option is a gain and
delay, not a complete measured frequency-dependent transfer function.

The manufacturer reports Qm=80 and dielectric loss tangent 0.015 for NCE51.
These quantities are stored as metadata but do not supply a unique broadband
causal loss model for the bonded PZT. They cannot be treated as measurements of
this experiment. Missing losses are a plausible contributor to waveform phase,
ringing and magnitude differences, not an established cause of the observed error.
The nominal free-disk planar resonance estimate Np/diameter is 192.5 kHz for a
10 mm NCE51 disk. Thus the tested band is not safely assumed to be far below all
disk resonances; bonding changes that response.

## Verification performed

Ran the existing focused suite:

    python -m unittest tests.test_pzt_receiver tests.test_pzt_body tests.test_piezoelectric_reference tests.test_pzt_bond

All 23 checks passed. They cover constitutive form equivalence, uniform-field
charge, mass, reciprocity, passivity and charge/power balance, independent RC ODE
agreement, rigid motions, tied-face affine continuity, open-circuit backreaction,
joint mass inversion, bond spring/mass invariants and spatial receiver integration.
Some receiver tests exercise the historical planar helper; these do not replace
the coupled component and body tests or verify the whole GPU run.

Existing refinement evidence changes body radial resonance by less than 1% and
sampled attached driving-point responses by less than 2% under successive body
refinements. Those are exploratory thresholds on a fixed small host patch, not
full-plate phase or waveform convergence. Existing full-size startup checks cover
128 steps only. No full simulation or new GPU benchmark was launched here.

The earlier author-simulation cross-check supports S0 packet agreement at 200 kHz
only, conditional on the author's supplied time vector being in milliseconds.
It does not establish all-frequency or A0 physical validation.

The STEMINC SM412 profile for the user's future 7 mm disks has only a reduced
planar description and explicitly lacks a complete 3D tensor. It cannot silently
replace NCE51 in this 3D queue. The audited Wandowski run uses 10 mm NCE51 disks.

## Sources

- COMSOL, governing equations and constitutive conversion:
  https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_theory.06.043.html
- Manufacturer Noliac Ver1404 complete NCE51 tables:
  https://www.mmech.com/images/stories/Technologies/Piezo%20Ceramics/Noliac_CEramics_NCE_datasheet_1404.pdf
- Current manufacturer NCE51 summary:
  https://www.ctscorp.com/Product-Series/NCE51.htm
- COMSOL, mechanical/electrical/coupling losses and associated phase lag:
  https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.109.html
- Wandowski experimental paper and public preprint:
  https://zenodo.org/records/18937842
- Local refinement evidence: `docs/evidence/pzt-body-readiness-refinement-final.json`.

Further physical validation needs independent measurements or a documented
transfer function for the actual bonded disks/readout, plus full-plate waveform
refinement. Tuning unknown properties to these traces alone would be calibration
to those traces and would require separate held-out validation.
