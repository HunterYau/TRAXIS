# Hardware assumptions and numerical controls

The student-authored
[Research Plan 2026-2027](https://docs.google.com/document/d/1juFE1R_Tuu8gbiNZUyFEOGWNv-KlCkcUIIZx4HFtyIU)
was rechecked on November 8, 2026. Its materials list proposes six 6061-T6
500 x 500 x 3 mm plates, one intact and five defective. Flat-bottom recess
diameters are 20/30/40 mm and depths 0.75/1.50/2.00 mm. These are proposed
specifications, not independently measured manufactured specimens.

The materials list says twelve PZT discs, about 7 mm in diameter, on an
approximately 350 mm circular array. The physical assembly procedure says
eight transducers at corners and edge midpoints. This unresolved difference
affects final coordinates and acquisition counts. The plan lists an AWG,
optional power amplifier, oscilloscope, relay matrix, shielded cables, epoxy
and a support fixture, without a final electrical/material calibration.

## Current verification controls

The full-size enriched temporal study deliberately retains its existing
3 mm-radius mechanical load/probe disks, one source at (20,20) mm and two
receivers at (250,250)/(480,480) mm. The disk diameter is 6 mm, distinct from
the plan's approximate 7 mm PZT proposal. It is an intact numerical control,
not the finalized sensor array. Changing radius or coordinates during the
time study would change the experiment rather than isolate temporal error.

The excitation uses a nominal 1 N normal-force scale at 100 kHz, with two Hann
cycles. Reception is area-average normal displacement. These quantities are
not a measured voltage drive or a calibrated voltage output. The model
retains finite free edges; physical supports/contact have not been matched.

## Required before physical/data acceptance

Record actual transducer part numbers, diameters and thicknesses, electrode
geometry, relevant material/damping data or a measured equivalent, adhesive
thickness/properties, actual sensor coordinates, drive amplitude/response,
receiver impedance/filtering, and support/contact conditions. Preserve the
separate intact/calibration/final-holdout plate roles in the confirmed study
plan. Do not infer missing values from a generic disc or from software checks.

These decisions remain open. The numerical study's fixed controls are
explicitly provisional; no current result accepts a PZT-voltage training bank.
