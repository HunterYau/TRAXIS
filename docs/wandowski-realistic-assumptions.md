# Realistic assumed setup for the Wandowski comparison (v9)

This is a predeclared plausible laboratory scenario, not a reconstruction of
unreported settings or an amplitude/time fit. It has not been shown to improve
experimental agreement. No full experiments were run for this update.

## Values and evidence

- Adhesive: finite elastic interface, E=1 GPa, nu=0.30, density=1100 kg/m3,
  thickness=100 um. These cyanoacrylate-like values were used by Eiras et al.,
  Sensors 24 (2024) 450, section 2.3: https://doi.org/10.3390/s24020450 .
  Wandowski identifies cyanoacrylate, but not those numerical properties.
- Receiver: assume the NI PXIe-5105 high-impedance setting, 1 Mohm parallel
  to nominal 50 pF. Its other 50-ohm setting is available, not selected here.
  Manufacturer specifications, page 3:
  https://assets-us-01.kc-usercontent.com/ecb176a6-5a2e-0000-8943-84491e5fc8d1/d70de596-1860-416b-9c51-d88ff65a9923/NI-PXIe-5105_datasheet.pdf .
- Cable: assume 1 m of RG58-like cable, 93.5 pF/m, based on Belden 9201:
  https://catalog.belden.com/techdata/en/9201_techdata.pdf . Cable type and
  length are assumptions. Total added capacitance is 143.5 pF per receiver.
- Drive: use the stored generator waveform without peak normalization,
  multiplied by assumed X10 gain. The Krohn-Hite 7500 manual lists X10,
  X100 and variable gain, but the experiment does not report which was used.
  A 0.2778 us effective delay approximates its specified trend to 100 degrees
  of phase lag at 1 MHz. This is not a measured transfer function; flat gain,
  no clipping and ideal prescribed output voltage remain approximations.
  https://krohn-hite.com/wp-content/uploads/2025/08/7500_man.pdf .
- Damping: retain the documented 6061-T6 narrowband proxy. The actual plate
  alloy is unidentified. Nominal material values are not called optimistic
  or replaced with arbitrary weaker values.

## Connected calculations

The bond uses shear stiffness G/h and constrained-normal stiffness
E*(1-nu)/((1+nu)*(1-2nu)*h), integrated with area-lumped contact weights.
Displacement jumps are allowed. Adhesive mass is lumped equally to each
endpoint of the zero-thickness interface. This is a thin-interface model,
not a resolved glue volume; it omits geometric offset, adhesive damping,
nonuniform coverage and debonding. PZT consistent mass remains unchanged.
There is no arbitrary sensor coupling multiplier.

Receiver input/cable capacitance enters the mechanical stiffness and voltage
probe. Free electrode charge evolves with finite resistance, using exact
RC relaxation at fixed displacement in symmetric half-steps. Its voltage
and mechanical force feed back into each other. The small-system GPU check
compares the coupled transient with an independent SciPy ODE solve for open,
1 Mohm and 1 kohm loads and checks time refinement and passive energy decay.
This is not just a filter applied to an already finished voltage trace.

All assumed settings are copied into each result and printed in the console.
New NPZs use simulated_receiver_voltage_v and
assumed_applied_actuator_voltage_v; the generator and unit-peak reference
are also saved for transparency. The unit-peak reference is not the drive.
Comparisons still expose raw voltage differences as well as normalized
outline/wiggle metrics. No received signal is used to select the assumptions.

## Remaining uncertainty

Support/contact locations, alloy, actual amplifier setting and transfer,
actual cable/termination/filtering, glue coverage/loss, wrap-around electrode
geometry, solder mass, specimen-specific losses and sensor polarity remain
unresolved. No fixture or clamp was invented. A free plate is still assumed.
The uncertainty scenarios listed in the setup profile are not additional
scheduled experiments. V9 remains exploratory, not accepted training data.
The original body-mesh refinements used a perfect bond; they do not establish
convergence for this new finite-interface model. Full plate/bond/load waveform
convergence and agreement with experiment remain to be established.

## Validation and scheduling

CPU interface checks cover rigid motion, added adhesive mass and analytical
uniform-slip energy. GPU receiver checks cover coupled consistent mass,
finite resistance, forced motion, free decay and second-order time refinement.
A separate CUDA regression preserves zero-port damped/undamped behavior.
The full geometry startup check runs 128 steps, only 0.684 us; it does not
validate the 350 us observation interval or agreement with the real data.

The new five-frequency queue is v9: 50, 100, 150, 200, 250 kHz in order.
Results from v6/v7 and v8 preparation/checks remain in their original folders.
Runtime resource stops remain disabled as requested; numerical/provenance
and saved-output checks remain. Cooling pauses remain enabled.
