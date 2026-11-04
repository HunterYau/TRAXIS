# Waveform bank and partial-observation contract

One bank represents one specimen scan. Every row is an explicitly ordered
transmitter/receiver/frequency action; columns are uniformly spaced waveform
samples. The bank includes aligned intact baseline traces, plate/sensor
geometry, a depth label in mm and provenance. A 512 x 512 reference label is
recommended; smaller grids are permitted for software checks.

Cache files are NPZ archives loaded with pickle disabled. Their manifest
records schema version, shape-independent action identities, metadata and
SHA-256 array checksums. Loading has an uncompressed byte limit. Saving uses
a temporary file and atomic publication; replacing a bank requires explicit
`overwrite=True`. Checksums detect corruption, not maliciously rewritten data.

`specimen_id` identifies a scan; `parent_specimen_id` identifies the physical
plate or complete simulated defect scenario. All variants/repeated scans of
one parent must share one split. `validate_specimen_splits` checks this across
a proposed dataset. Simulation uses train/validation/test. Physical specimens
use calibration/adaptation/physical_holdout; final holdouts cannot be adapted.
The project still needs a reviewed dataset manifest and enforcement in the
training/evaluation entrypoints. A metadata field alone does not prove that
an external source was physically validated.

`ReplayEnvironment.state()` and `.reveal(index)` return only geometry, action
identities, the time grid, acquired waveforms/baselines, and a Boolean observed
mask. Missing records are zero-filled; observable units remain public. Labels, defect parameters, specimen
identifiers and solver settings are not present. Copies prevent modification
of a returned state from changing the bank/environment. This is a software
interface, not protection against Python code inspecting private attributes.

## External solver / physical measurement import

`python -m traxis import-traces --manifest bundle/manifest.json --output data/scan.npz`

Prepare one scalar CSV for each current/baseline trace. Its first line must
be exactly `time_s,value`. Times are seconds; values use the declared observable
(`displacement_m`, `velocity_m_s`, `voltage_v` or `arbitrary`). Export the same
time grid for every record; the importer refuses silent interpolation or unit
conversion. Mechanical displacement is not PZT voltage. Record reference-solver
mesh/time/source/support/bonding settings and validation status honestly.

An example manifest for a one-record demonstration bundle:

```json
{
  "schema_version": 1,
  "metadata": {
    "specimen_id": "sim-0001-scan-1", "parent_specimen_id": "sim-0001",
    "split": "train", "domain": "simulation", "observable": "displacement_m",
    "baseline_specimen_id": "intact-reference-1", "solver": "external FEM export",
    "validation_status": "plumbing_only", "seed": 123,
    "settings": {"model_revision": "record the solver/model revision here"}
  },
  "plate": {"width_m": 0.5, "height_m": 0.5, "thickness_m": 0.003},
  "sensors_xy_m": [[0.02, 0.02], [0.48, 0.48]],
  "records": [{
    "action": {"transmitter": 0, "receiver": 1, "frequency_hz": 100000.0},
    "waveform_csv": "current-0-1-100k.csv", "baseline_csv": "intact-0-1-100k.csv"
  }],
  "depth_mm_npy": "geometry-depth-mm.npy"
}
```

Bundle paths stay within the manifest directory. Source file and manifest
hashes are added to provenance. Exporting a commercial solver's native binary
format or connecting to hardware is not implemented by this importer.

## Imaging and metrics

The DAS baseline subtracts the intact signal, takes its analytic envelope and
samples it at bistatic group delays, with an explicit time offset for the
drive packet/acquisition system. Mean evidence over acquired records is
returned with a timing-coverage map. Set mode/frequency-specific group speeds
from control measurements or an explicitly chosen wave model. No bulk speed
fallback, per-trace peak normalization or unselected defect signal is used.
This follows the geometric model in
[Sharif Khodaei & Aliabadi (2014)](https://doi.org/10.1088/0964-1726/23/7/075007).
Straight paths and one dominant mode are approximations, especially at a
deep recess; DAS intensity is a localization index, not depth in mm.

Metrics compare cell-average predictions on a shared physical grid. Integer
refinement replicates coarse cells; integer coarsening averages blocks.
Whole-plate and true defect-region errors are both reported. Threshold-based
overlap, centroid, equivalent diameter, false-positive area and loss volume
carry explicit units. A missed defect has no centroid error value and is
flagged separately; an intact plate has no relative loss-volume error.
Two empty masks have Dice/IoU=1. Thresholds must be fixed using training or
validation data before final evaluation. Ensemble spread/coverage diagnostics
do not assert confidence calibration.
