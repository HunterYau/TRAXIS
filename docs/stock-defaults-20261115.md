# Prospective stock defaults

New `Plate()` instances, unspecialized scenario planning, model depth bounds
and the primary dispersion CLI now use 3.175 mm stock. The study recipe
already specified this thickness and the requested PZT variation. Its
debug architecture configuration now matches the same nominal thickness.
The model-input thickness-mismatch guard remains unchanged.

Historical tensor, GPU tensor and recess geometry/shape references explicitly
retain their recorded 3 mm thickness. Five tensor case fingerprints were
captured before changing the default; all five and the fourteen embedded
case/screen/profile identities remain unchanged. Complete old scenario and
bank records retain their explicit thickness and signals when loaded.
Recorded geometry readers now require width, height and thickness, instead
of applying current planning defaults to incomplete saved data. Incomplete
bank or external-trace geometry rejects before loading signal arrays/files.

Integration testing caught two migration problems. Existing model defaults
still used a 3 mm bound, which correctly triggered the retained mismatch
guard against new 3.175 mm fixtures. The isolated test process also spawned
some children through the main-checkout portable interpreter, producing a
source-dependency mismatch. Matching the model defaults and using the isolated
CPU runtime corrected these issues. Four subsequent test failures were old
3 mm assertions for map bounds, interval clipping and normalized RL reward;
those assertions now use the actual specimen thickness. The failed test
reports remain available alongside the passing result.

The final CPU discovery run completed 454 tests: 419 passed and 35 optional
checks were skipped, with no failures or errors. It includes bounded neural
updates, replay, exact checkpoint recovery, specimen/split protections,
architecture comparisons and uncertainty workflows. These are software
fixtures; no research training cycle or physical performance claim follows.
Six small coupled-layer GPU force/interface checks also pass, with exact
mass and maximum force discrepancy 2.005e-16. They stop at batch 16 and
do not allocate full-size XYZ operators or complete XYZ windows.

A fresh two-version CUDA transition bundle retains zero differences.
Historical time/scattering reviews still pass time refinement and retain
the original 1.344761% spatial-scattering failure. Old receipts are not
rewritten. Existing source-bound artifacts still need their appropriate
review/validator chain; preserving a physical case does not silently rebind
an old certificate to new source hashes.

Current compact evidence: evidence/stock-default-migration-20261115.json.
The fresh isolated bundle is
`outputs/stock-defaults-source-transitions-20261115-v1/source-transitions-check.json`,
fingerprint 8389682077da5dedfc166d0de18b9921d94a0629cb64048a263399dd4aa844f2.
The live main-checkout control and its original CPU check were kept on
their bound sources through completion before any integration.

The pilot-bank audit also identified a separate unfinished correction:
its intact CAD seed uses the defect scenario's geometry seed, and its
shared automatic time step is selected using the damaged operator. Those
choices can imprint defect-dependent numerical information into a baseline.
Correct baseline independence before promoting any bank. Current pilot
banks remain `plumbing_only`; full XYZ convergence, representative broader
wave cases, calibrated PZT response and physical validation remain open.
