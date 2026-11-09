# Full-plate layered spatial controls

The enriched candidate needs independent in-plane and thickness checks.
The first layered family keeps a 500 x 500 x 3 mm intact box and 5 mm x/y cells,
with one, two or four thickness layers. All cells use the same six-tetrahedron
body-diagonal pattern. The plate and 500-microsecond observation window are
not reduced for these research controls. The orientation stays fixed within
this family; it does not establish general directional accuracy.

| Cells | Tetrahedra | Enriched scalar / vector coefficients |
| --- | ---: | ---: |
| 100 x 100 x 1 | 60,000 | 321,603 / 964,809 |
| 100 x 100 x 2 | 120,000 | 582,805 / 1,748,415 |
| 100 x 100 x 4 | 240,000 | 1,105,209 / 3,315,627 |

`examples.layered_mesh` screens exact counts and geometry buffers before
allocation. It vectorizes the same conforming pattern as the independently
tested legacy generator, then checks box volume, actual enriched topology,
shared-face multiplicity and boundary faces. It writes a geometry-only bundle:
no native space, CUDA operator, sparse matrix or wave solve is constructed.
The bundle explicitly records those limits and binds its source, specification,
mesh/report hashes and process-tree resource observations.

The forward driver accepts `--mesh-bundle` or the existing `--capacity` input,
with distinct checked protocols. On load it reconstructs the declared layered
geometry and compares saved coordinates/connectivity exactly before CUDA work.
The independent saved-state checker supports both inputs. Historical wave
plans remain tied to their original driver snapshots/revisions; the prior
time-convergence experiment is preserved on `research/enriched-time-refinement`.
The numerical `traxis` implementation is unchanged by these input extensions.

Thirteen selected checks pass without skips, including a bounded small GPU
bundle-to-waveform run and independent CPU final-state verification. Pure tests
match geometry/counts against the legacy construction, reject limits before
coordinate allocation, and detect changed bundle data. These are input-path
software checks, not physical validation of the large meshes.

The declared wave settings use the same 100 kHz/two-cycle force scale, 3 mm
source disk and two diagonal receivers, plus a third fixed off-axis receiver
at (350,200) mm. All meshes in the new study use these same three footprints.
The extra receiver checks a different propagation direction; it is not a
replacement for the unresolved physical sensor layout.

```powershell
.\.venv\Scripts\python.exe -m examples.layered_mesh --spec configs/layered-100x100x1.json --output outputs/layered-mesh-z1
.\.venv\Scripts\python.exe -m examples.enriched_gpu_wave --mesh-bundle outputs/layered-mesh-z1 --settings configs/enriched-layered-wave.json --mode operator --output outputs/layered-operator-z1 --max-seconds 240 --max-rss-mib 1536
```

Actual full-size GPU capacity/CFL and complete-waveform spatial comparisons
remain to be measured at this commit. A common stable time grid will be frozen
before propagation. Full-size in-plane refinement, defects/frequencies, PZT
coupling/calibration and physical agreement remain necessary. No training bank
is accepted and no full training ran.
