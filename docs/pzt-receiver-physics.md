# PZT charge and voltage receiver implementation

The previous five-frequency benchmark is preserved at commit `a234ce2` on
`archive/five-frequency-benchmark-a234ce2` and
`research/experimental-five-tests`. New work is isolated on
`research/pzt-receiver-physics`. No benchmark result, timing window or
experimental waveform has been changed. No new full experimental plate
simulation or training run was performed.

## Implemented response

`traxis.piezoelectric` implements small-signal linear piezoelectricity with
engineering Voigt order xx, yy, zz, yz, xz, xy and poling along axis 3:

```
strain = sE @ stress + d.T @ E
D = d @ stress + epsilonT @ E

cE = inverse(sE)
e = d @ cE
epsilonS = epsilonT - d @ cE @ d.T
stress = cE @ strain - e.T @ E
D = e @ strain + epsilonS @ E
```

Inputs must be finite, real, symmetric where appropriate, and physically
positive. A profile with incomplete full tensors cannot silently become a
3D material. Polarity and electrical boundary conditions are explicit.

The bounded 3D tetrahedral component assembles anisotropic stiffness,
consistent mass, electromechanical coupling and dielectric operators.
Grounded/equipotential electrodes and charge-free side boundaries condense
to a mechanical/electrical port:

```
M u'' + Ksc u + g V = F
Q_top = Cp V - g.T u
```

The convention is V=phi_top-phi_bottom. It includes mechanical inertia and
electrical backreaction; its harmonic response supports passive resistive
and capacitive loads. Attachment constraints are supplied by the caller.
No salol layer, mechanical loss, electrode mass or catalog-Q damping is
invented. Dense assembly is intentionally bounded to small PZT components,
not a full plate. Coupling a resolved PZT body into the large plate mesh is
still separate work.

`traxis.pzt_receiver` provides an efficient adapter to the existing tensor
plate solver. It integrates both in-plane strain directions and in-plane
shear over the actual circular area. A boundary integral of displacement
gives the same finite-element strain charge, retaining spatial cancellation.
Its sparse probe coefficients have units C/m; multiplying plate displacement
returns charge in C. They plug into `sparse_probe_set` and the existing CUDA
explicit sampler. The legacy sampler's array field is named
`displacement_m`; when using these probe weights its contents are charge,
and must be labelled `strain_charge_c` by the caller, as in the check example.

The efficient adapter is a **one-way, perfectly bonded, quasistatic
plane-stress observer**. It does not add PZT mass/stiffness or reaction
forces to the plate operator. Its reported ceramic mass is informational.
It does not reproduce radial resonances, bond slip or frequency-dependent
losses. It is not a fully coupled physical-hardware simulation or accepted
training-data generator. Near 300 kHz, the proposed STEMINC disc specifically
requires dynamic characterization rather than trusting this approximation.

The passive readout solves, for piecewise-linear strain charge:

```
(Cp + Cinput) Vdot + V/Rinput = q_strain_dot
```

It returns voltage, electrode/input-capacitor charge, drained charge and a
charge-balance residual. Cable capacitance belongs in Cinput. None for
resistance explicitly means an ideal open input; it does not mean an
unspecified real oscilloscope. A short circuit has V=0 and Q_top=-q_strain.
No ADC conversion, gain, noise or transfer function is fabricated.

## Sourced material profiles

### NCE51 reference for full 3D checks

[Noliac Ver1404](https://www.mmech.com/images/stories/Technologies/Piezo%20Ceramics/Noliac_CEramics_NCE_datasheet_1404.pdf),
printed pages 9-11, provides a complete sE, d and free-permittivity set.
The original tables were downloaded, hashed and visually inspected. The
profile stores SI values and derives cE, e and epsilonS from that one set.

The overview d31=-195 pC/N differs from the full-matrix -208 pC/N. The
separately printed e33=17.2 C/m2 also differs from the 15.90798 C/m2 derived
from the selected independent set. Do not mix those tables. Derived
epsilonS33/epsilon0=822.50269 closely agrees with the separately listed 823.
[Current CTS specifications](https://www.ctscorp.com/Product-Series/NCE51.htm)
corroborate d31=-208 pC/N, d33=443 pC/N, free dielectric constant 1900 and
density 7850 kg/m3, but are not mixed into the older full tensor.
These are nominal small-signal reference values, not measured coefficients
of the experimental specimens. The source's tolerances and conditions are
retained in the JSON profile.

### User's proposed STEMINC part

[SMD07T02S412 product page](https://www.steminc.com/PZT/en/piezo-ceramic-disc-7x02mm-300-khz)
and [SM412 material table](https://www.steminc.com/piezo/PZ_property.asp)
confirm:

- 7 mm diameter, 0.2 mm thickness, silver electrodes on opposite faces.
- Radial resonance 300 +/-10 kHz; this is not a thickness resonance or a
  hard frequency cutoff. Bonding may change the response.
- Static capacitance 3000 pF +/-15%, measured at 1 kHz/1 Vrms.
- Product kp>=0.55 and resonant impedance <=10 ohms are bounds.
- SM412 nominal material kp=0.63, d31=-190 pC/N, epsilonT33/epsilon0=1850,
  density 7800 kg/m3.

The public table does not provide all coefficients required for a 3D SM412
tensor. The reduced profile therefore uses the ideal planar identity
kp^2=2*d31^2/((s11+s12)*epsilonT33), deriving the compliance sum and
plane-stress coupling. This is explicitly a reduced reference model, not
a measured full tensor. Generic PZT-5A and NCE51 coefficients are not used
to fill the missing entries.

The material-based free capacitance is about 3.152 nF, within the device's
listed static-capacitance tolerance. The derived in-plane-constrained
capacitance is about 1.901 nF; these are different mechanical boundary
conditions. The 3 nF product value is not blindly substituted for every
capacitance in the coupled equations. Catalog Q and 1 kHz loss tangent are
recorded but not extrapolated as broadband damping.

## Hardware intent and unresolved inputs

`configs/pzt/user-hardware-plan.json` records the user's eight proposed
SMD07T02S412 sensors on 3.175 mm 6061-T6 aluminum, probably using salol
(phenyl salicylate). Sensor positions/polarities, salol thickness and
mechanical/loss properties, generator, drive voltage, scope/DAQ input and
cable capacitance are unknown. Null means unknown, not zero, open circuit
or perfect bonding. Historical 5 mm mechanical patches remain historical;
this part's radius is 3.5 mm. Do not overwrite literature-test geometry with
the proposed hardware.

Published salol usage exists, for example in
[this ultrasonic transducer experiment](https://www.mdpi.com/1996-1944/14/9/2269).
That does not establish the properties or thickness of the user's bond.
For full hardware fidelity, obtain the missing SM412 tensor or independent
admittance/strain-to-voltage measurements, characterize the attachment, and
specify the acquisition electronics. Do not fit the old FBG timing peaks.

## Usage and verification

```python
from traxis.piezoelectric import read_profile, planar_from_profile, Readout
from traxis.pzt_receiver import DiskGeometry, disk_receiver

profile = read_profile('configs/pzt/steminc-smd07t02s412.json')
material = planar_from_profile(profile)
# Illustration only: supply the experiment's real centre, surface and polarity.
geometry = DiskGeometry((0.25, 0.25), 0.0035, 0.0002, 'top', 1)
receiver = disk_receiver(layout, material, geometry)
probes = receiver.probes()  # use with the plate CUDA sampler
# q_c is the sampled integrated strain charge, not old FBG voltage/line strain.
readout = Readout(resistance_ohm=None, capacitance_f=0.)  # explicit IDEAL input
record = receiver.record(time_s, q_c, readout)
voltage_v = record['voltage_v']
```

Do not relabel the saved FBG traces as PZT measurements. A PZT benchmark
must use the actual PZT receiver channels, matching location and electronics.
Changing a gain or choosing another packet peak is not validation.

Focused tests verify constitutive identities, rejection of incomplete/nonpassive
data, signed electrode charge, open/loaded circuits, electrical power balance,
rigid-motion rejection, both strain directions, shear, polarity, rotated
anisotropic coupling, and circular spatial cancellation against an analytical
Bessel integral. The new bounded example compares 3D operators independently
with NGSolve and compares CUDA charge sampling against exact CPU normal modes.

```
python -m unittest tests.test_pzt_receiver tests.test_piezoelectric_reference -v
python -m examples.pzt_receiver_checks --output outputs/new-pzt-check --native --gpu
```

These are software/component checks, not additional real-data trials or proof
that the prior 100/150 kHz mismatch is fixed. The five saved results remain
unchanged. Full material/adhesive/electrode/electronics validation is still open.

Final verification: 438 CPU tests passed, 35 optional checks skipped, zero
failures/errors (473 total). The 22 focused checks pass. Independent native
3D operator discrepancies are at most 2.42e-14 relative; passive coupled
energy balance is within 1.88e-14 relative. The CUDA observer agrees with
exact CPU normal modes to about 1.56e-11 relative, and its final charge
matches direct CPU evaluation to about 1e-15. These small manufactured
fixtures verify implementation, not real hardware accuracy.

The full CPU suite peaked at 61.4% system RAM and 813.42 MiB owned-process
RSS. Detailed evidence, source hashes and limitations are recorded in
[the implementation receipt](evidence/pzt-receiver-implementation-checks.json).
The first redirected GPU-check command returned a PowerShell error status
despite its completed proof; a repeat launched through Python with combined
stdout/stderr explicitly returned exit code 0 and completed=true. The
original v1/v2 artifacts are retained rather than rewritten.
