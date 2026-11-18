# Experimental reference search, November 17, 2026

Later update: the user requested PZT-body physics. That implementation and a
bounded full-size execution check are now complete. A fresh five-frequency
scheduler is prepared but held for mesh refinement/reference selection; no
full experimental batch started. See `pzt-body-checkpoint-20261117.md`.
The research-stage description below predates that implementation.

No new simulation was launched. The requested list is 50, 100, 150, 200,
250 kHz, with 300 kHz only if matching experimental data are available.
The machine-readable file `configs/pzt/next-experiment-plan.json` is a
pending plan, **not an executable or running scheduler**. A replacement
experiment has not yet passed selection. This does not establish that no
suitable experiment exists; it records the limits of this search.

## Most relevant primary sources

| Reference | Useful information | Why it does not yet replace the current experiment |
| --- | --- | --- |
| [Yang et al., Scientific Data 2025](https://doi.org/10.1038/s41597-025-05300-5), [author repository](https://github.com/SmartDATA-Lab/Long_Term_Guided_Waves) | 530 x 530 x 3 mm aluminum, eight STEMINC SM412 discs, 7 mm x 0.2 mm, sensor coordinates, equipment and environmental records; public waveform repository | Uses a 1 ms 5–350 kHz chirp, not the requested separate tone bursts. Alloy and measured bond properties remain unspecified in reviewed material. Outdoor support/environment and protective epoxy add reconstruction uncertainty. Authors explicitly describe reflections obscuring the first arrival, including after pulse compression. |
| [Perfetto et al., Materials 2021](https://www.mdpi.com/1996-1944/14/24/7602) | 6061 plate, 287 x 287 x 2 mm, PIC255 sensors, adhesive identified, excitation/equipment and FEM description | Data availability says “Not applicable”; no downloadable raw recordings located. Chirp excitation; reported sensor-size wording and timing need clarification. Naming an adhesive does not supply measured bond thickness. |
| [Ahmed and Kopsaftopoulos, Sensors 2021](https://www.mdpi.com/1424-8220/21/16/5672) | 6061 plate, PZT-5A dimensions and spacing, Hysol EA9394, acquisition and numerical setup | Data availability says “Not applicable”; no raw waveform download located. Study's 200–700 kHz range does not provide 50–150 kHz reference measurements. Printed dimensions contain unit-conversion inconsistencies. |
| [FEM-Based Modeling of Guided Acoustic Waves, Applied Sciences 2025](https://www.mdpi.com/2076-3417/15/16/9116) | Plate and PIC255 actuator geometry; experimental frequency sweep and scanning measurements | Receiver is laser vibrometry, not PZT voltage. It cannot directly validate the requested PZT recording behavior. |
| [Santoni-Bottai and Giurgiutiu, AIAA 2012, author PDF](https://research.cec.sc.edu/files/lamss/files/j86_aiaa_2012_santoni-bottai_exact_shear_lag.pdf) | 2024-T3 plate, 1200 x 1060 x 1 mm; 250 mm sensor spacing; three-cycle bursts, equipment and tuning curves | Bond thickness is varied in the prediction; this must not be mistaken for a measured specimen bond. Frequency sweep uses 20 kHz increments starting at 10 kHz. Raw recordings and all requested frequency cases were not established. |
| [Wandowski et al., existing dataset](https://zenodo.org/records/18955122), [author manuscript](https://zenodo.org/records/18937842), [published paper](https://doi.org/10.1088/1361-665X/ae0d1d) | Actual drive and both PZT receiver recordings; plate and NCE51 dimensions, coordinates, equipment; plate 1 covers 50–250 kHz | Still the most directly usable of the reviewed datasets for these five frequencies. Alloy, glue mechanics, exact instrument/cable transfer and some installation details remain unknown. Plate 2's 300 kHz data must not be silently merged into plate 1. Existing batch is already complete; an unchanged rerun would not resolve the mismatch. |

The Utah repository now points to a Dropbox replacement because its original
Figshare link is unavailable. The README and published setup were reviewed;
the large monthly pickle files were not downloaded or executed. Similar
ceramic composition and dimensions do not establish that its packaged sensor
is identical to the user's proposed SMD07T02S412.

Other searches included Open Guided Waves benchmarks (primarily CFRP),
6061-T6 author studies and theses, laser measurements, and damage/localization
datasets. None was promoted to ground truth merely because it contains the
words aluminum, experimental, or PZT. Numerical datasets are not physical
measurements. No source was contacted and no author was emailed.

## Model limitations independent of dataset choice

The existing plate run uses a reciprocal planar actuator and area-integrated
PZT strain-to-charge receivers. It does not put the PZT body's mass and
stiffness, adhesive mechanics, or radial resonance into the full plate model.
The small coupled piezoelectric component is not yet integrated into the
large plate solver. A better paper alone cannot remove these omissions.
Missing values must stay unknown, or be explicitly labeled assumptions in
an exploratory sensitivity test; they must not be represented as measured.

The previous paired analysis remains valid as a saved comparison, not proof
of full validation: fast arrival errors approximately 1.3, 4.1, 1.3, 5.3,
5.7 percent at the five frequencies; slow arrivals at 150–250 kHz remain
tentative because they exceed the paper's reported PZT A0 detection range.
See `pzt-paired-comparison-20261117.md` for the full evidence.

## Display and next action

`python -m examples.pzt_terminal --resources --debug` now adds whole-machine
RAM usage/free memory and NVIDIA GPU utilization, device memory and temperature.
Shared GPU memory is explicitly not measured. Worker log locations are shown;
this is a progress/log display, not an attached breakpoint debugger.
The viewer is read-only and cannot start or stop simulations. Existing
completed results must not be displayed as new running tests.

The user was asked whether to hold for better measurements or prepare another
explicitly approximate comparison using the existing experiment. Pending that
choice, preserve all completed outputs. Before any new launch, prepare a fresh
source-bound queue with a meaningful controlled change and automatic comparison;
do not simply repeat identical runs or label a plan as scheduled execution.
