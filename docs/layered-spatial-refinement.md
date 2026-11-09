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

## Measured geometry and GPU capacity

All three complete geometry bundles pass volume, topology, shared-face and
boundary checks. Actual full-size native coefficient maps, positive mass,
box moments and GPU/CPU operator comparisons also pass. No time propagation
runs in these operator measurements.

| Thickness layers | Stable-step bound (ns) | GPU/CPU relative action error | Peak combined RSS (MiB) | Monitored seconds |
| --- | ---: | ---: | ---: | ---: |
| 1 | 63.1488 | 3.75e-15 | 505.09 | 38.80 |
| 2 | 35.6466 | 7.88e-15 | 612.50 | 74.39 |
| 4 | 19.0740 | 2.14e-14 | 827.00 | 143.54 |

The largest actual space has 3,315,627 vector coefficients. Its operator fits
inside the 1.5 GiB process-tree cap; sampled system-wide GPU memory is 3,366 MiB
and temperature 54 C. Device storage does not include a global or dense local
stiffness matrix. These capacity results do not establish wave convergence.

## Frozen waveform protocol

`configs/layered-thickness-study.json` declares the three full-window cases
before propagation. The common step count is at least 65,536, rounded up to
a power of two if the smallest measured stability bound needs more. The
maximum CFL fraction is 0.4 on every mesh. Per-case fractions are derived to
produce exactly the same integer step count and sample times; no interpolation
or shortened window is allowed. All forcing/receiver settings and numerical
sources remain fixed.

The provisional thickness discrepancy screen requires a decreasing raw L2
difference and a finest relative change below 5% for each receiver separately.
It is not a continuum error bound. Frozen settings, operator receipts,
runtime/GPU identities and source snapshots are written by:

```powershell
.\.venv\Scripts\python.exe -m examples.layered_spatial_study prepare --output outputs/layered-thickness-study-20261108
```

Each complete waveform must have its independent CPU saved-state check before
assessment. The existing thermal pacer can watch each explicit project plan;
wall caps are configurable while the 78/72 C pacing band and original 80 C
stop remain unchanged. The largest wave case permits 2 GiB combined RSS for
its larger force/probe/state arrays, while system RAM remains capped at 85%
with at least 2 GiB available. No allocation is implied by a higher cap.

Fourteen selected input/state/pacing/study checks pass without skips. The final
three pure study checks additionally cover empty/complex records and confirm
the fixed common-grid and 5% criteria. The last complete integrated suite is
the earlier 268-test result; selected counts are not added to it.

Complete-waveform thickness comparisons remain pending at this commit.
Full-size in-plane refinement, defects/frequencies, PZT coupling/calibration
and physical agreement remain necessary. No training bank is accepted and
no full training ran.
