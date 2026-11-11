# Current physical specimen and mounting requirements

The user revised the hardware requirements on November 11, 2026:

- Nominal aluminum thickness is 1/8 inch = 3.175 mm. Measure actual stock
  thickness for calibration; nominal sheet size is not a measured tolerance.
- PZTs will be removable and clamped against the plate, rather than epoxy bonded.
- Placement and contact will vary between mounting sessions. The simulator
  and evaluation must account for these variations.

`configs/scenarios-study.json` now specifies 0.003175 m for new research
scenario recipes. Its sensor layout, frequency list and existing nuisance
ranges remain provisional. Existing 3 mm configurations, reference fixtures
and measurements retain their original settings and provenance. In particular,
the legacy `Plate()` default and tensor reference protocols still specify
3 mm; new physical-study work must use an explicit 3.175 mm plate. Migration
of those prospective workflows and fresh numerical verification are required.

For unchanged recess depths 0.75/1.50/2.00 mm, remaining stock becomes
2.425/1.675/1.175 mm. Guided modes, mesh/time requirements and waveforms need
checking at the revised thickness. Earlier 3 mm convergence is historical
evidence, not acceptance of the revised specimen.

## Clamp mounting and repeatability

Clamped contact must not be treated as a perfectly bonded epoxy interface.
Record clamp geometry, preload, contact footprint, plate support and whether
a couplant is present. Clamp mass/stiffness and local contact may influence
the plate response as well as signal amplitude. Dry versus couplant-assisted
contact and PZT construction are not yet resolved.

Represent measured placement offsets per mounting, changes in effective
contact footprint and pressure, transmitter/receiver coupling, and possible
frequency/phase response changes. Do not invent calibrated jitter/contact
ranges: estimate them from repeated removal/reclamping on the intact plate.
Store actual coordinates separately from nominal coordinates available to
the reconstruction, so robustness tests do not silently reveal exact placement.

The current scenario code varies scalar sensor gains and gain drift. This is
an amplitude nuisance model, not a validated clamped-contact model, and it
does not yet perturb sensor locations or footprint shape. Those extensions
and independent contact/physical verification remain implementation work.
All mounting repeats of a physical plate must remain grouped by parent plate
when splitting data. Distinguish paired measurements without remounting from
baseline/current measurements after reclamping.

Contact-ultrasound guidance describes the role of couplant and consistent
pressure; see Evident's [couplant explanation](https://ims.evidentscientific.com/en/applications/ultrasonic-couplant)
and [coupling-technique discussion](https://ims.evidentscientific.com/en/learn/ndt-tutorials/thickness-gauge/factors).
These are general contact-testing references, not validation of the planned
bare-PZT clamp assembly. No waveform bank or training is accepted here.
