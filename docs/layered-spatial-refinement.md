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

The compact capacity record is `evidence/layered-capacity-debug.json`, with
fingerprint `a679a64ae406ed83573d0e0cc8023aaa842c6322debde440cdb137fe342c8f2e`.
Its 44 file identities were rechecked before recording this milestone.
It covers geometry/operator capacity, not complete-waveform convergence.

## Completed windows and thickness screen

The frozen one-, two- and four-layer runs each complete 65,536 increments and save
65,537 samples through the full 500 microseconds. The common increment is
7.62939453125 ns. No source, material, forcing, receiver or runtime setting
changes between these levels; their geometry and stability-derived CFL
fractions differ as declared in the study plan.

| Thickness layers | Propagation / monitored seconds | Peak combined RSS (MiB) | Postburst modified-energy range |
| --- | ---: | ---: | ---: |
| 1 | 357.815 / 400.954 | 612.67 | 1.04e-15 |
| 2 | 515.291 / 597.379 | 738.62 | 6.87e-16 |
| 4 | 1,041.389 / 1,189.113 | 982.02 | 8.59e-16 |

All three independent CPU saved-state checks pass, with exactly matching
positive mass. Maximum relative energy discrepancies are 4.72e-14, 1.68e-13
and 4.08e-13; final-probe discrepancies are at most 1.62e-16. These checks independently
apply streamed stiffness to the saved final displacement, rather than replaying
every step on the CPU. The pacing receipts complete normally with no pauses.
Raw records are under `outputs/layered-wave-z1-20261108`,
`outputs/layered-wave-z2-20261108`, `outputs/layered-wave-z4-20261108` and
the corresponding `layered-cpu-z1/z2/z4` and `layered-pacing-z1/z2/z4`
directories. Resumed November 9 jobs keep the
original frozen family suffix.

The completed comparison **fails** the unchanged 5% per-receiver screen:

| Receiver (mm) | Raw 1-to-2-layer difference (m, discrete L2) | Raw 2-to-4-layer difference (m, discrete L2) | Finest relative change |
| --- | ---: | ---: | ---: |
| (250,250) | 3.72262e-8 | 5.60305e-9 | 18.0010% |
| (480,480) | 5.11458e-8 | 4.56243e-9 | 9.90936% |
| (350,200) | 3.37357e-8 | 5.09953e-9 | 19.5625% |

Every successive difference decreases, but none meets the finest-change limit.
The complete comparison retains all 65,537 common raw samples and their late
edge reflections. Similar peak amplitudes and conserved discrete energy do
not establish thickness accuracy. No screen, waveform window or receiver set
was changed after observing this result.

The four-layer run peaks at 75 C, 54.2% system RAM, at least 7,435.71 MiB
available RAM and 1,423 MiB sampled system-wide GPU memory. Its explicit
workspace screen is 406.33 MiB. It completes 65,537 stiffness applications,
zero per-step full-state host transfers and two final downloads. No cooling
pause occurs. CPU temperature remains unavailable. All three wave workers,
pacers and state checks have normal terminal receipts.

The assessment is `outputs/layered-thickness-assessment-20261109`.
`evidence/layered-thickness-debug.json` binds complete raw wave/state/pacing
records, settings, source identities, resource summaries and receiver metrics;
fingerprint `a6d63f180a2336b71d92ddca89a72b3e48284acb5a3e79bc87ce81dcad825ed8`.
Its collector independently checks the comparison against original traces and
exports PNG/PDF plots under `outputs/layered-thickness-tools-20261109`.
The PNG was visually checked for complete time span, units, labels and clipping.

Further thickness accuracy and its computational cost must be resolved.
Full-size in-plane refinement, defects/frequencies, PZT coupling/calibration
and physical agreement remain necessary. No training bank is accepted and
no full training ran.
