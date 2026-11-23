# Does the analytic pulse reasonably represent the excitation?

## Conclusion

Yes, as a representation of the experiment's specified ideal generator pulse.
It is not an independently calibrated representation of the voltage actually
delivered to the PZT after the generator, amplifier, wiring and capacitive load.
This review used stored data and calculations only; no simulation was launched
and no production/default settings were changed.

## Evidence

- The Wandowski paper specifies a five-cycle sinusoidal toneburst with a Hann
  (called Hanning in the paper) envelope. The analytic formula implements exactly
  that definition. At 250 kHz the pulse lasts 20 microseconds.
- Using the public stored excitation samples at 50, 100, 150, 200 and 250 kHz,
  the unnormalized formula agrees at every stored time to within 3.5e-15 V.
  No fit of frequency, phase, delay or gain was used to obtain that match.
- Both old and new runs divide the source by the same recorded sample peak.
  This preserves the original sampled normalization, rather than adding a fitted
  gain. It does not calibrate the experiment's actuator voltage.
- At 250 kHz the normalized continuous formula peaks at approximately 1.00049 V,
  versus 1 V at the maximum stored sample. That 0.049% difference is the expected
  distinction between the continuous peak and a sampled peak. It was not removed
  by another normalization or a fit.
- The source voltage and its first two derivatives join continuously to zero at
  the pulse endpoints. Evaluating the formula removes artificial corners between
  stored samples while preserving the stated frequency, cycles, phase and envelope.
- The saved source samples in the completed trial were checked against the
  implemented analytic function and matched exactly. The receiver waveforms and
  final states have recorded checksums.

The exact integral of normalized source voltage squared over time is 1.6801%
higher at 250 kHz than for straight-line reconstruction. This reflects the small
attenuation introduced by the old interpolation. It is not a measurement of
mechanical energy imparted to the plate or electrical energy delivered without
a load/current model. Across 50–250 kHz the calculated difference ranges from
0.0667% to 1.6801%.

The controlled full-system runs independently showed that changing only source
evaluation greatly suppresses the previously observed high-frequency ringing,
and that the raw A0-related temporal difference becomes 0.0752%. Those outcomes
support improved numerical source representation; they do not establish actual
experimental voltage calibration.

## Physical limits

The PXIe-5413 can perform interpolation/reconstruction filtering, so straight
segments between data points should not be assumed to be its exact analog output.
Its actual configuration and analog output were not established from the stored
excitation samples alone. Those samples may describe the commanded waveform;
their exact mathematical agreement does not establish an independent terminal
voltage measurement.

The experiment's Krohn-Hite 7500 has a documented frequency-dependent gain and
phase response. Its actual gain setting, load-dependent response and voltage
across the bonded PZT were not identified here. The trial continues the earlier
ideal normalized-voltage assumption and does not simulate those unknown effects.
PZT mechanical feedback is present in the coupled solver, but the actuator port
still prescribes an ideal voltage, without a calibrated source impedance.

Consequently, the method reasonably represents the intended excitation shape.
Actual actuator strength, electronics delay and distortion remain unvalidated.
Exact terminal reproduction would need independent voltage/current or transfer
measurements; those quantities should not be fitted merely to improve receiver
agreement.

## Sources and artifacts

- Original experiment and preprint: https://zenodo.org/records/18937842
- Public data and description: https://zenodo.org/records/18955122
- NI generator specifications:
  https://www.ni.com/docs/en-US/bundle/pxie-5413-specs/page/specs.html
- NI reconstruction/filtering explanation:
  https://www.ni.com/en/shop/electronic-test-instrumentation/waveform-generators/generating-signals-function-generators.html
- Amplifier manufacturer manual:
  https://krohn-hite.com/wp-content/uploads/2025/08/7500_man.pdf
- Numerical check record:
  `outputs/result-analyses/physical-excitation-check-20261122/source-check.json`
- Source implementation: `examples/pzt_analytic_source.py`
- Completed controlled trials: `outputs/pzt-development/analytic-drive-ab-20261122/`
