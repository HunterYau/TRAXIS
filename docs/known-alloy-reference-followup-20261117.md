# Known-alloy reference follow-up

The user requested another search for a physical experiment with substantially
more complete properties, especially a known aluminum alloy. No simulation
was run. No replacement configuration was selected and the frozen v3 queue
was not changed. Having more recordings does not establish that a setup is
better characterized.

## Most promising new material match

[Amarel et al., author manuscript, section III and Data Availability](https://arxiv.org/html/2409.06084v3):
6061-T6, 610 mm square, 1.2 mm thick; four APC Model 63 transducers in a
350 mm square. Five-cycle Hann bursts at 300 kHz, 0.4 ms recordings; six
undamaged baselines and 2601 contact-load examples. Public compressed data;
raw recordings require an author request. This improves alloy identification,
but does not supply the requested five-frequency validation. Exact PZT geometry,
bond mechanics, support and instrument details remain unverified. This is a
candidate, not an approved replacement or an exact reconstruction.

[Author repository](https://github.com/USNavalResearchLaboratory/MINC) was
inspected through its README, file tree and processing/reading code. Four
processed JLD2 files are listed (approximately 35, 36, 48 and 67 MB). The
loader names separate baseline and damaged arrays. Its processing subtracts
means, selects Fourier bins, inverse-transforms and discards outliers. The
CSV reader explicitly remaps sensor ordering. None of the binary data was
downloaded or decoded; no Julia code was executed or installed.

Before treating its compressed data as a waveform benchmark, inspect the
actual time/channel metadata and reproduce its exact processing on simulated
signals. Compressed inputs cannot simply be compared sample-for-sample with
unprocessed voltages. Requesting raw data would also allow timing/trigger,
drive amplitude and calibration questions to be resolved, subject to what
the authors actually retained. No authors were contacted.

## Other checked candidates and limits

| Primary source | Finding and decision |
| --- | --- |
| [NASA/ASU fatigue lap-joint dataset](https://catalog.data.gov/dataset/fatigue-crack-growth-in-aluminum-lap-joint) | Physical PZT actuator/receiver signals and optical crack lengths. The riveted joint/fatigue setup adds contact and geometry requirements; no equivalence to a single free plate was established. Keep as a later, more complex validation candidate. |
| [COQTEL public dataset](https://zenodo.org/records/14193336) | Two physical corrosion datasets in HDF5, but the specimen is stainless 316L steel. Do not mistake the broader project's aluminum work for the material in this downloadable dataset. |
| [UGW propagation on cracked AL 2024-T3 plate](https://data.mendeley.com/datasets/f3kd5d6mw5/2) | Known alloy and numerous signals, but the dataset description explicitly identifies simulated stochastic-FEM signals. It cannot serve as independent real-world truth. |
| [Lamb wave behavior at different severity of cracks](https://data.mendeley.com/datasets/xywmdn7x9g/1) | Also simulation data, generated using Abaqus Explicit. Excluded as real-world ground truth. |
| [Ono, experimental Lamb-wave attenuation](https://www.mdpi.com/2076-3417/12/13/6735) | Includes 6061-T6, measured attenuation and calibrated displacement sensing. Useful separate damping evidence; the reviewed supplement is a parameter spreadsheet, not a raw PZT-to-PZT burst dataset. Different thickness and contact sensors prevent direct substitution. |
| [Ahmed and Kopsaftopoulos](https://pmc.ncbi.nlm.nih.gov/articles/PMC8402505/) | Known 6061, identified PZT-5A geometry, adhesive and acquisition. Still no public raw download located; studied 200–700 kHz and analyzed 250 kHz. Printed metric/imperial dimensions disagree, requiring clarification. |

## Outcome and next useful step

No candidate checked satisfies both near-complete reconstruction information
and verified measured recordings across 50–250 kHz. This is a limit of this
search, not evidence that such a dataset does not exist. Do not fill missing
values from another experiment and claim they describe this specimen.

The NRL candidate merits a targeted request for unprocessed baseline/drive
records, PZT part drawing/material constants, adhesive and bond thickness,
supports, temperature, equipment settings and electrical loads. It would be
a separate 300 kHz reference if those gaps can be resolved. A known alloy
alone cannot resolve PZT/bond/instrument uncertainty.

Correction to the preceding chat explanation: the current reference's
71 GPa, 2700 kg/m3 and 0.33 are the nominal values retained in our existing
reference setup. They are not measured properties of its unidentified alloy.
An alloy label is useful, but measured elastic properties, density and
attenuation of the actual plate would constrain wave behavior more directly.
