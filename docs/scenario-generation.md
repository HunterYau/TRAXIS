# Virtual specimens and local elastic banks

Scenario recipes separate specimen geometry, acquisition settings and nuisance
variation. They are reproducible inputs, not completed simulations. Every
local waveform bank remains `plumbing_only` and `ready_for_training_data=false`.
No large synthetic dataset or research training has been run.

## Commands

Write recipes without solving wave physics:

```powershell
.\.venv\Scripts\python.exe -m traxis plan-scenarios --config configs/scenarios-study.json --output outputs/study-recipes.json
```

Use `--variants 2` to create two nuisance realizations per parent while keeping
each parent's split. The CLI caps planning at 10,000 total recipe rows; it does
not allocate one waveform bank per row. `parent_count` and `parent_split_counts`
distinguish independent virtual parents from nuisance/repeat rows.

Export one selected recipe as an unsolved external reference task:

```powershell
.\.venv\Scripts\python.exe -m traxis export-solver-request --plan outputs/study-recipes.json --index 0 --output outputs/reference-task
```

This writes `solver-task.json`, a unit-explicit Gmsh/OpenCASCADE GEO script,
512-grid geometry labels, requested sample times and an importer-compatible
CSV mapping. It supplies no fake zero traces and launches no commercial solver.
The exported GEO parses to the correct CAD volume in tests. Gmsh coordinate
numbers are assigned meters by the recipe; any downstream CAD import must
set units explicitly. PZT/adhesive/electrical properties required for voltage
simulation are listed as missing inputs, not fabricated defaults.

For a small local pilot, use the explicitly different 40 mm geometry profile:

```powershell
.\.venv\Scripts\python.exe -m traxis plan-scenarios --config configs/scenarios-pilot.json --output outputs/pilot-recipes.json
.\.venv\Scripts\python.exe -m traxis generate-bank --plan outputs/pilot-recipes.json --index 0 --settings configs/forward-pilot.json --cache-dir outputs/elastic-groups --output outputs/intact-bank.npz
```

The local command generates only one selected specimen. It refuses to overwrite
an existing bank/report. Pilot output is still unsuitable for research training.
Reference exports anchor label and requested-time file hashes; trace import
rejects changed labels or output time grids rather than merely accepting a
different uniformly sampled record with plausible metadata.

## Reproducible study recipes

The default sampler describes 1,000 virtual parents on 500 x 500 x 3 mm plates.
It includes 10% intact cases and balances the nine combinations of 20/30/40 mm
diameter and 0.75/1.50/2.00 mm backside loss among damaged parents. Centers vary
while preserving configured edge and sensor-footprint clearance. Geometry and
nuisance seeds are independent and recorded.

Recipes include nominal 6061-T6 elastic constants with provisional variation,
equivalent force amplitude, per-sensor coupling gains, current/control coupling
drift, Gaussian signal noise, sensor layout, frequency catalog, pulse cycles,
sampling and support description. These ranges are computational assumptions,
not measured distributions. The local solver supports only free boundaries
and refuses other support descriptions. Their study requires a suitable
reference solver and measured support/coupling properties.

Eight corner/edge and twelve ring PZT layouts are explicit alternatives. Their
reciprocal five-frequency catalogs contain 140/330 records. Directed catalogs
contain 280/660. Reciprocity remains an experimental assumption to check.
Hann-burst sampling is screened using its first main-lobe zeros; it is not a
claim that spectral tails are absent or that the DAQ filter is modeled.

Parent IDs and split assignments stay fixed under nuisance variants. Separate
virtual parents receive deterministic approximate 70/15/15 split assignments;
the saved plan reports actual split and diameter/depth counts. Small cohorts
can have missing strata/splits and must be reviewed. Multiple virtual intact
parents represent separate material/noise realizations, not repeated physical
measurements claimed as independent plates. Physical scans use the explicit
parent-role workflow instead.

For uncertainty calibration, the study config reserves an expected 10% of all
parents from the training allocation, with validation/final-test identities
unchanged. The sampler's default reservation is zero; it is an explicit study
setting. Reserved parents receive `uncertainty_calibration`, retain that role
under variants, and cannot be used for model or policy selection. See
uncertainty-calibration.md for the single-scan protocol and assumptions.

Plan files are fingerprinted and declare themselves unsolved. A recipe does
not certify mesh convergence, PZT voltage fidelity or physical agreement.

## CAD-conforming recess mesh

The optional Gmsh/OpenCASCADE backend subtracts circular cylinders from the
backside of the solid plate, leaving the top surface intact. CAD volume is
compared with the linear-tetrahedron volume. Curved boundaries are still
approximated by facets; the measured relative material-loss volume error is
recorded and a configurable error gate applies before solving.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-meshing.txt
```

The backend runs headlessly with one mesh thread, no user config files and no
GUI. It screens a conservative element/assembly-memory estimate before native
meshing, and checks actual element count afterward. It refuses to disturb an
already initialized Gmsh API session. Native meshing is not preemptible by the
cooperative resource guard. Gmsh and its Python package are GPL-licensed; see
[the Gmsh project](https://gmsh.info/) and [API manual](https://gmsh.info/doc/texinfo/).

The original structured-cell backend remains available for regression tests.
It rejects recesses whose volume error exceeds the configured limit instead
of silently producing a label for unresolved material loss. Correct volume
alone does not prove correct footprint, wave dispersion or scattering.

## Local pilot physics and cache

`generate_waveform_bank` assembles the same 3D isotropic elastic operator used
in the numerical verification tests. Its source is a unit-total-force normal
top-surface patch, and its probes are area-averaged normal displacement in
meters. There are no PZT, adhesive, electrodes, voltage readout or damping
domains. Coupling/drift is represented only by linear scalar gains. This is
an inspectable mechanical pilot, not a calibrated bonded-PZT forward model.

One transmitter/frequency solve records all receiving patches simultaneously.
A paired intact and defective solve use the same material, source and coupling
before explicit current gain drift/noise. This matched synthetic baseline is
not a separately measured intact physical plate; baseline mismatch must be
validated before real-data inference. Intact controls reuse their single system.

The output sample interval is divided into a whole number of conservative
stable internal steps. Probe traces are subsampled on that common aligned
grid, never silently interpolated. Steps, source settings, meshes, dependencies,
scenario and source hashes are retained in bank provenance. Noise is seeded
per action and current/control channel so cached recovery is deterministic.

Raw unit-load transmitter/frequency groups are atomically cached with integrity
checks. Drive/coupling/noise-only changes can reuse those raw solves when the
material/geometry/mesh/time settings match. Corrupt caches fail rather than
being overwritten. Pilot limits screen solve count, matrix memory, bank bytes,
internal steps and geometry. A fine full-plate request fails the local resource
screen; it is not launched just because a recipe contains 500 mm geometry.

Required next checks remain frequency-band/through-thickness resolution,
finite-plate mesh/time convergence, sensor/source fidelity, reflections,
supports and intact physical agreement. The independent Rayleigh-Lamb tests
already show that low-order bending resolution can be expensive; a cleaner
CAD mesh does not remove that limitation. COMSOL remains the preferred
reference if the required license is available.
