# Current physical specimen and PZT variation requirements

The user revised the hardware requirements on November 11, 2026:

- Nominal aluminum thickness is 1/8 inch = 3.175 mm. Measure actual stock
  thickness for calibration; nominal sheet size is not a measured tolerance.
- Placement and effective coupling will vary between trials. The simulator
  and evaluation must account for these variations.

On November 12 the user removed clamp calculations from the requested scope.
Keep slight, reproducible PZT position, footprint and coupling randomization;
do not make clamp geometry, preload, stiffness, friction or contact-pressure
calculations a development or acceptance requirement. This instruction
supersedes the earlier removable-clamp modeling requirement. It does not
specify a replacement mounting method or measured coupling law.

`configs/scenarios-study.json` now specifies 0.003175 m for new research
scenario recipes. Its sensor layout, frequency list and existing nuisance
ranges remain provisional. Existing 3 mm configurations, reference fixtures
and measurements retain their original settings and provenance. `Plate()`
and neural depth-bound defaults now use 3.175 mm. Historical tensor and
geometry references explicitly retain 3 mm. Saved scenarios, banks and
imported traces require complete recorded plate dimensions and preserve
their stored thickness. See stock-defaults-20261115.md for migration checks.

For unchanged recess depths 0.75/1.50/2.00 mm, remaining stock becomes
2.425/1.675/1.175 mm. Guided modes, mesh/time requirements and waveforms need
checking at the revised thickness. Earlier 3 mm convergence is historical
evidence, not acceptance of the revised specimen.

## PZT repeatability

Represent measured placement offsets per mounting, changes in effective
contact footprint, and transmitter/receiver coupling. Estimate jitter and
coupling ranges from repeated measurements on the intact plate when available.
Frequency/phase response should be added only with a defined response model
and evidence for its range; scalar gain is not a substitute for that response.
Store actual coordinates separately from nominal coordinates available to
the reconstruction, so robustness tests do not silently reveal exact placement.

The scenario code now varies positions, effective circular contact radii,
coupling gains and drift, with independent baseline/current remounts available.
See pzt-domain-randomization.md for provisional ranges and solver behavior.
These are mechanical robustness proxies. PZT/electrical response and physical
agreement still need independent verification; clamp mechanics are out of scope.
All mounting repeats of a physical plate must remain grouped by parent plate
when splitting data. Distinguish paired measurements without remounting from
baseline/current measurements after repositioning. No waveform bank or
training is accepted by this requirements update.
