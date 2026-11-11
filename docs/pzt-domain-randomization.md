# PZT mounting domain randomization

New research recipes use the 3.175 mm plate and seeded mounting variation.
These are mechanical robustness proxies for removable clamps, not a calibrated
contact, couplant, friction or piezoelectric model.

## Provisional study ranges

`configs/scenarios-study.json` enables:

| Quantity | Sampling rule |
| --- | --- |
| Position | Uniform in a disk of radius 0.5 mm around each nominal center |
| Effective contact radius | Uniform from 0.9 to 1.1 times nominal |
| Coupling strength | Independent per-sensor gain from 0.9 to 1.1 per mounting |
| Current gain drift | Additional factor from 0.98 to 1.02 |
| Drive-force scale | Existing uniform factor from 0.9 to 1.1 |
| Signal noise | Existing Gaussian noise at 1% baseline RMS |

These are initial sensitivity settings, not measured tolerances. Radius
variation implies roughly -19% to +21% contact-area variation. The patch stays
circular; rotating it would not change its footprint. Tilt, elliptical or
uneven contact, clamp mass/preload/stiffness and frequency-dependent coupling
still need measured models. Scalar gain is not identified contact stiffness.

The study enables `remount_between_states`: intact baseline and current state
draw independent positions, contact radii and coupling strengths. An intact
plate can consequently have a nonzero baseline-subtracted signal without
damage. Set the flag false for matched numerical controls. Coupling drift in
remount mode includes the ratio of new to baseline coupling, followed by the
small additional drift; its combined value need not be within 2% of one.

A mounting stays fixed across all transmitter/receiver/frequency records in
its session. `--variants` creates new remount realizations while keeping the
plate, recesses, material, parent ID and split fixed. Material varies across
parents. A separate seeded mounting stream means that changing the coupling
range does not change position/contact draws.

## Physical ports, caches and model inputs

`PZTMounting` stores actual offsets and contact-radius scales.
`Scenario.sensor_geometry(state)` resolves physical centers/radii. Sampling
rejects edge crossings, overlapping PZT-equivalent disks and configured
defect-clearance violations; it does not clip positions. Rejection stops after
128 attempts. Defect placement reserves the full mounting envelope so later
remounts retain clearance.

The local P1 and native P2/high-order waveform paths integrate loads/probes at
the actual centers/radii. Changed positions or radii invalidate raw-solve
cache keys. Gain/drive/noise-only changes may reuse unit-force solves.
An intact remounted state needs distinct load/probe solves even though its
material operator is unchanged. Scalar gains apply on both transmitter and
receiver sides, preserving the assumed reciprocal pair response.

Banks expose nominal coordinates to reconstruction, DAS and policy inputs.
Actual coordinates/radii stay in simulator provenance. This tests unmeasured
placement error; measured-coordinate inputs should be a separate experiment.
External requests export per-state actual ports and gains; the importer does
not apply gains/noise, so follow the force/readout instructions once.
Fixed-port tensor/mapped benchmarks remain separate verification fixtures;
they are not automatically converted into randomized waveform banks.

## Reproducibility and use

Library defaults keep position/contact variation zero. The research config
opts in explicitly. Legacy nominal scenario fingerprints are preserved.
Nominal plans remain schema 1; mounting plans and external solver tasks use
schema 2. The plan loader accepts both; old readers should reject schema 2
instead of ignoring its ports. The trace-import manifest retains its existing
schema because its coordinates intentionally describe nominal inference inputs.

```powershell
.\.venv\Scripts\python.exe -m traxis plan-scenarios --config configs/scenarios-study.json --count 8 --variants 3 --output outputs/clamp-recipes.json
.\.venv\Scripts\python.exe -m traxis export-solver-request --plan outputs/clamp-recipes.json --index 0 --output outputs/clamp-request
```

These write unsolved recipes/tasks, not training data. All remounts of a
parent, including calibration parents, retain that parent's split. Repeated
physical mountings must not become independent train/test plates.

Estimate ranges by repeated removal/reclamping on an intact plate, recording
position, clamp setting/preload, couplant condition and waveform repeatability.
Keep final physical holdouts protected when selecting those ranges. No clamp
contact fidelity or research-bank acceptance is implied by these proxies.

## Verification checkpoint

The final guarded integration passes 50 tests without skips in 35.286 test
seconds. It includes a native P2 pulse at 3.175 mm with changed current-state
footprints, legacy fingerprint/plan compatibility, both sensor layouts,
clearance rejection, parent/material/split preservation, real waveform/cache
effects and existing refinement tamper checks. Peak process-tree RSS is
414.99 MiB, system RAM 54.7%, minimum available RAM 7,348.28 MiB and GPU
observation 951 MiB / 48 C. CPU temperature is unavailable. No training ran.

The first invalid-footprint test expected the wrong error wording; the
nominal edge check correctly rejected it earlier. A broader run also caught
two old producer fixtures using raw dataclass serialization. They now use the
canonical scenario serializer; strict source/geometry mismatch checks remain.
Both failed logs and successful intermediate/final runs remain retained.
Overlapping test groups are not added to the older integrated test total.

Retain a small waveform sensitivity reference with:

```powershell
.\.venv\Scripts\python.exe -m examples.verify_pzt_randomization --output outputs/mount-reference
```

The saved 40-by-40-by-3.175 mm intact control changes one sensor by (0.4,-0.3)
mm and increases its effective contact radius by 10%. Combined waveform L2
change is 5.4931%; an independent current remount produces the same nonzero
intact residual. This is a mechanical sensitivity example, not a prediction
of real clamp variability. Gain-only changes reuse all seven raw groups and
match the expected scaling within 1.71e-8 relative. Position/contact changes
require fresh raw solves. The fixture remains plumbing-only.

A separate recipe audit checks 24 realizations from eight parents, 384
baseline/current sensor positions, preserved parent splits/materials and a
schema-2 external task with 70 jobs / 140 records. The final PNG/PDF figure
was visually checked. Raw artifacts are under outputs/pzt-randomization-*.
Compact evidence is evidence/pzt-domain-randomization-debug.json, fingerprint
7511c071096d772ae63cf3fde1e6c187a1c76836f1108231412d0a94436f3720.
