# Why the high-frequency ringing changes between runs

## Scope

This investigation uses existing saved waveforms, source inspection, closed-form
calculations and published references. No new simulation, eigenmode solve, mesh
study, test run configuration, or scheduler was created or started. The user
requested approval before setting up or running new simulations. Production
physics, drive selection, numerical step sizes and the stopped queue are unchanged.

Analysis script: `examples/analyze_high_frequency_cause.py`.
Machine-readable evidence and source checksums:
`outputs/result-analyses/high-frequency-cause-20261122/cause-analysis.json`.

## Strong evidence for the inconsistency: temporal numerical dispersion

The late top-minus-bottom receiver spectrum (150–350 microseconds) contains a
dominant peak whose position shifts with the integration time step:

| Case | Step size | Observed dominant peak | Inferred continuous-time mesh-system frequency |
| --- | --- | --- | --- |
| original | 5.3406 ns | 5.235856 MHz | 5.229124 MHz |
| half dt | 2.6703 ns | 5.230832 MHz | 5.229153 MHz |
| quarter dt | 1.3351 ns | 5.229580 MHz | 5.229161 MHz |

The frequency shifts shrink by a factor of 4.0136 when dt is halved. This is
consistent with second-order temporal dispersion, despite the full-waveform
percentage appearing to stall. The current lossless system uses velocity Verlet.
For an unforced linear mesh eigenmode its exact discrete frequency relation is

    f_numerical = asin(pi * dt * f_mesh) / (pi * dt)

Equivalently,

    f_mesh = sin(pi * dt * f_numerical) / (pi * dt)

Using the finest trace's inferred frequency as a diagnostic reference predicts
the other two peaks to about 37 Hz and 8 Hz, respectively. Other nearby spectral
peaks show a similar collapse under the inverse relation. This does not identify
the mode's physical origin: f_mesh is the frequency of the discretized spatial
system, not an independently established true resonance.

At the dominant frequency the predicted excess phase accumulated over 350
microseconds is approximately 2.356, 0.5875 and 0.1468 cycles for the three step
sizes. These are substantial phase shifts even at the smallest dt. Thus a raw
waveform L2 difference can remain large while modal frequencies are converging:
the phases are not yet close enough for the small-error fourfold waveform rule
to apply to this component. Numerical stability and energy conservation do not
guarantee phase accuracy.

Peak locations use Hann-window spectra, zero padding and quadratic interpolation
of log power. These improve estimation but do not add information beyond the
200-microsecond observation window. Close modes and nonstationary components can
bias the estimates. This is strong consistency evidence, not an exact modal solve
or a fitted correction applied to validation waveforms.

Reference: Olver and Stern, *Dispersive Fractalization in Linear and Nonlinear
Fermi–Pasta–Ulam–Tsingou Lattices*, section 4.1, equations 4.1–4.2:
https://par.nsf.gov/servlets/purl/10257293

## Strong lead for excitation: straight-line drive reconstruction

Every stored generator signal checked (50, 100, 150, 200 and 250 kHz) matches
the published five-cycle Hann-windowed sine definition at its stored sample times:

    v(t) = sin(2*pi*f*t) * sin(pi*t/T)^2, 0 <= t <= T
    T = 5/f; v(t) = 0 outside the pulse

Maximum point differences are below 3.5e-15 V; relative L2 differences are below
1.9e-15. No amplitude, phase, timing or received-signal fit was needed to obtain
that match. The simulator currently joins these samples with straight lines
instead of evaluating their known continuous definition.

On a fixed analysis grid, the 250 kHz straight-line waveform has power in
5.1–5.4 MHz at -52.60 dB relative to the 150–350 kHz carrier band. Evaluating the
continuous source definition gives -138.97 dB in the same calculation. The
approximately 86 dB difference is an input spectrum calculation only; it is not
a demonstrated reduction of receiver ringing.

Interpolation images appear near the stored input sample rate plus/minus the
carrier. Existing late receiver peaks also tend to follow the upper image band:
about 5.053 MHz for 50 kHz input, 5.105 MHz for 100 kHz, 5.178 MHz for 150 kHz,
5.201 MHz for 200 kHz and 5.236 MHz for 250 kHz. A fixed set of structural modes
excited with different strengths can also produce this trend. It supports the
source-image hypothesis but does not establish causality without changing the
source in an otherwise identical simulation.

NI describes interpolation and reconstruction filtering for its waveform
generators, and the PXIe-5413 specifications distinguish image rejection with the
digital filter enabled from operation without that rejection. The experiment's
actual filter settings and PZT-terminal waveform are still unknown. Analytic
source evaluation would fix representation of the stated ideal source, not
establish the actual generator/amplifier response.

https://www.ni.com/docs/en-US/bundle/pxie-5413-specs/page/specs.html
https://www.ni.com/en/shop/electronic-test-instrumentation/waveform-generators/generating-signals-function-generators.html
Original experiment: https://zenodo.org/records/18937842

## Other candidates

| Candidate | Evidence and current conclusion |
| --- | --- |
| Aliasing when comparing with the 5 MS/s measurements | Real processing risk, addressed by the previously added separately labelled anti-aliased comparison. The troublesome peak already exists in raw saved simulation data at 23.406 MS/s (Nyquist 11.703 MHz). Therefore downsampling to 5 MS/s did not create this raw peak or its raw convergence difference. It can fold the peak toward 230 kHz afterward. |
| PZT thickness resonance | Manufacturer thickness frequency constant 2000 Hz m gives a nominal 4.0 MHz scale for a 0.5 mm disk. Constrained bulk estimates from the implemented tensor set are about 3.912 MHz (short) and 4.442 MHz (open). These are not 5.23 MHz and are not the bonded assembly's eigenfrequencies. Thickness/other PZT resonance remains plausible, but the peak cannot currently be identified as a physical thickness mode. |
| Spatial numerical dispersion / underresolution | The Q5 plate's maximum in-plane node spacing is 1.141 mm. At 5.229 MHz, longitudinal and shear bulk reference wavelengths are about 1.194 mm and 0.601 mm. This is a substantial high-frequency resolution warning. The current mesh was selected for low-frequency guided waves; full 5 MHz spatial convergence has not been shown. Fixed mesh does not explain dt-dependent frequency shifts by itself, but can set inaccurate high-frequency mesh modes. |
| Lossless PZT and plate | These runs have no active material damping or resistive readout. Unwanted ringing is not dissipated. Real PZT has mechanical/electrical losses, but a catalog Q or loss tangent does not uniquely establish broadband bonded-device damping. Arbitrary damping would risk hiding the numerical problem. |
| Top/bottom subtraction | It leaves a much smaller residual than the dominant common signal and can magnify relative differences. This explains sensitivity of the A0-related score; it does not create the observed MHz frequency shift. The residual is not proof of a pure A0 mode. |
| Reflection and mode interference | Full recordings contain multiple arrivals; late portions dominate the disagreement. They can affect L2 scores and spectral peak shapes, but do not remove the close agreement with the Verlet frequency relation. |
| Numerical blow-up | Saved post-pulse modified-energy changes are at floating-point precision, with finite bounded traces. There is no evidence of instability/exponential energy growth in the completed records. Energy conservation alone does not certify phase or spatial accuracy. |
| GPU reduction roundoff / nondeterminism | Cannot be fully excluded without matched repeated runs, but the systematic frequency shifts agreeing with the integrator relation are a stronger explanation than random numerical noise. No repeat simulation was performed. |
| Oscilloscope/amplifier/cable | They may explain disagreement with real data. They are not active varying parameters in the three ideal numerical runs and cannot independently explain those runs' dt-dependent peak shifts. |

NCE51 thickness constants and nominal material data:
https://www.mmech.com/images/stories/Technologies/Piezo%20Ceramics/Noliac_CEramics_NCE_datasheet_1404.pdf
PZT loss mechanisms:
https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.109.html
Digitizer specifications:
https://www.ni.com/docs/en-US/bundle/pxie-5105-specs/page/specs.html

## Proposed fix and approval-dependent validation

The first proposed change is to evaluate the exact five-cycle Hann toneburst
instead of linearly interpolating its stored samples. Use the same normalization
factor as the current recorded drive. Do not fit received amplitudes or phases,
modify PZT properties, add guessed damping, or use output filtering as proof that
the underlying raw ringing has disappeared.

Approval is needed before implementing the simulation variant or preparing its
run files, per the user's instruction. Proposed controlled validation:

1. A 250 kHz, 350-microsecond run at the original dt, changing only source
   reconstruction. Compare its raw high-frequency power with the existing baseline
   and confirm that the intended carrier-band response remains consistent with
   the known source change.
2. Repeat that new source at half dt. Compare the complete unfiltered voltage
   traces, isolated spectral bands and declared direct-arrival diagnostics.
   Check whether the raw A0-related difference falls below the existing 1% target.

Use newly separated output folders so the old convergence study remains intact.
Based on saved same-machine runs, approximately 45 minutes plus 90 minutes,
roughly 2–3 hours total; actual runtime depends on load and cooling. Both retain
the full current plate and coupled PZT bodies. Stop after the first run if its
outcome indicates that the second would not address the cause.

If high-frequency ringing remains, the next discriminating work is a targeted
modal/spatial study of the bonded PZT/plate system. That needs separate approval
and must precede asserting a physical thickness-resonance explanation or adopting
new high-frequency material/damping values. Smaller dt or a phase-accurate
integrator may then be needed for any high-frequency content that is physically
intended. None of these simulation studies has been set up or run here.
