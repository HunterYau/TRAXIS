# Directional P2 plate refinement

The measured P2 study below predates the [mass quadrature correction](elmer-mass-quadrature.md).
Its failed screens describe the archived formulation. Corrected directional
results require regeneration; none of these traces are approved training data.

The [unstructured finite-plate study](elmer-plate-refinement.md) failed its
spatial screen while passing its time screen. This follow-up varies in-plane
and through-thickness resolution separately on the same intact box, with
an additional independent time check. It preserves the earlier failure and
does not generate training data.

```powershell
.\.venv\Scripts\python.exe -m examples.guarded_directional_reference --plan outputs/waveform-control-plan.json --settings configs/elmer-directional-refinement.json --elmer-home outputs/reference-tools/elmer-26.1/ElmerFEM-nogui-nompi-Windows-AMD64 --cache-dir outputs/elmer-p2-directional-groups --output outputs/elmer-p2-directional-refinement --monitor-output outputs/elmer-p2-directional-monitor
.\.venv\Scripts\python.exe -m traxis plot-directional-waveforms --sweep outputs/elmer-p2-directional-refinement --records 0 3 --output outputs/elmer-p2-directional-refinement-figures
```

Create the fixed control plan with `plan-scenarios` and
`configs/waveform-control.json` as described in [elmer-waveforms.md](elmer-waveforms.md).
Use new output directories. The numerical reference is the same free
40 x 40 x 3 mm plate, nominal aluminum properties, eight radius-3 mm circular
apertures, TX 0, seven receivers, a two-cycle 100 kHz Hann source, and 128
samples at 0.5 microseconds. Mechanical normal force/displacement remains
an idealization; there are no PZT, adhesive or electrical domains.

## Mesh and independent axes

The intact box uses six linear tetrahedra per rectangular cell, with a common
body diagonal. ElmerGrid inserts straight edge midpoints for the P2 field.
Node/element/assembly screens run before mesh allocation. The mesh's measured
tetrahedron volumes must reproduce the exact box volume within 1e-12
relative error, and converted P2 node/element counts must match the screen.
`cells` controls this backend; `mesh_size_m` applies to the CAD backend.
Structured reference runs reject current geometries containing recesses.
Defect studies must use a conforming CAD mesh, rather than this grid's
centroid-removal approximation.

The plan freezes seven distinct cases before native execution:

| Axis | Cases | Held fixed |
| --- | --- | --- |
| In-plane | 10 x 10, 15 x 15, 20 x 20 cells | Four thickness cells, dt=31.25 ns |
| Thickness | 1, 2, 4 cells | 20 x 20 in-plane cells, dt=31.25 ns |
| Time | 125, 62.5, 31.25 ns | 20 x 20 x 4 cells |

The finest case is shared by all three axes. Circular P2 load/probe integrals
retain their physical area, center and second moments on every grid.
Signed coefficients are preserved. All comparisons use float64 displacement
on the original sample times, without amplitude fitting, time/phase shifts,
interpolation or removal of rigid motion/reflections.

Each adjacent pair must have resolved reference signals. The predeclared
maximum finest-pair relative L2 tolerances are 5% in-plane, 5% through the
thickness and 1% in time. Geometric thickness/time refinements also report
an observed change order. The in-plane levels are not geometric, so no
simple logarithmic order is inferred for that axis. These are discrepancy
screens, not continuum error bounds. Holding one spatial axis fixed can
leave error or interactions in that axis; a passing screen is not physical
acceptance.

## Integrity, cost and limits

Every case uses the same source/kernel/runtime/scenario/action checks as
the original refinement. Saved arrays must exactly match the compared
values. The three temporal cases must have identical mesh/aperture hashes.
Reports retain case inputs, source snapshots, solver/worker receipts,
discrepancies, energy/linear-residual diagnostics and resource samples.
PNG/SVG outputs include both the raw traces and differences from the shared
finest case for preselected records.

The example allows at most seven new groups and 600 seconds for the inner
study. Its outer process-tree monitor allows ten extra seconds for startup
and termination and caps combined monitor/descendant RSS at 1,280 MiB,
system RAM at 85%, and minimum available RAM at 2,048 MiB. GPU memory and
temperature guards remain active. The matrix assembly screen is explicitly
280 MiB to admit the 9,600-tetrahedron fine grid's conservative 263.7 MiB
estimate. Computation uses CPU, and CPU temperature is unavailable on this
host. No full training or large waveform cohort runs.

The initial 768 MiB attempt stopped on the finest system after completing
two grids. Recovery retained those inputs/results and changed only resource
bounds after measuring about 4,955 MiB available system memory. All physical
parameters, mesh/time levels, solver and accuracy thresholds were preserved.
The outer RAM/free-memory bounds are stricter than the inner CLI defaults.

Completed failed screens return exit code 2. Operational failures return 1
and preserve partial reports. After verifying the process has stopped, a
new output directory can reuse only completed checksum-checked groups from
the same cache. Recovery retains the failed attempt and its cost. Every
result remains `ready_for_training_data=false`, including passing screens.

## Measured directional control, 2026-11-04

| Axis and comparison | Maximum raw relative L2 change | Predeclared result |
| --- | --- | --- |
| In-plane: 10 to 15 cells per direction | 57.14% | Diagnostic pair |
| In-plane: 15 to 20 cells per direction | 12.83% | **Failed 5% screen** |
| Thickness: 1 to 2 cells | 41.60% | Diagnostic pair |
| Thickness: 2 to 4 cells | 8.01% | **Failed 5% screen** |
| Time: 125 to 62.5 ns | 0.8823% | Diagnostic pair |
| Time: 62.5 to 31.25 ns | 0.2205% | Passed 1% screen |

All seven receiver signals were resolved. Observed thickness and temporal
change orders were 2.36889 and 1.99909. Both spatial sequences decreased,
but their final changes still exceed the original tolerances. The tested
grids remain unsuitable for research training. These results identify
unresolved error along both spatial axes; they do not establish a single
cause for the earlier unstructured mesh changes or an absolute continuum
error bound.

| Structured grid | P2 nodes / degrees of freedom | Tetrahedra |
| --- | --- | --- |
| 10 x 10 x 4 | 3,969 / 11,907 | 2,400 |
| 15 x 15 x 4 | 8,649 / 25,947 | 5,400 |
| 20 x 20 x 4 | 15,129 / 45,387 | 9,600 |
| 20 x 20 x 1 | 5,043 / 15,129 | 2,400 |
| 20 x 20 x 2 | 8,405 / 25,215 | 4,800 |

The shared finest mesh has 0.75 mm maximum thickness span. Its direct
factor reported 65,620,378 stored entries. Across the completed new solves,
maximum relative linear residual was 1.92e-15 and postburst relative energy
range was 3.69e-14. The two cached cases retain their original solver
receipts. Raw/difference PNG and SVG figures for the preselected TX 0 to
RX 1/RX 4 records were visually checked, including units, legends and
provisional labels.

The completed recovery took 485.01 s inside the study and 486.33 s under
the outer monitor, using five new groups and two checked cache hits. Sampled
outer process-tree RSS peaked at 1,009.9 MiB, system RAM at 76.5%, and
available memory stayed above 3,809 MiB. System GPU readings peaked at
1,831/8,192 MiB and 48 C; the simulation used CPU. CPU temperature was
unavailable. The initial failed 768 MiB attempt and its cost remain recorded
separately in [evidence/elmer-directional-refinement-debug.json](evidence/elmer-directional-refinement-debug.json).

The default direct ordering and an isolated reverse Cuthill-McKee/LU probe
both exceeded the initial 768 MiB guard on the finest system. An algebraic
resource experiment used a scalar consistent-mass inverse as a
[conjugate-gradient preconditioner](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html).
It solved known-solution systems at all three time steps with at most
1.64e-11 relative solution error, using 19/35/66 iterations and 515.7 MiB
sampled combined RSS. Its earlier 512 MiB stop is also retained. This probe
measures algebraic accuracy and resource cost. All measured waveforms in
the directional study use the original direct symmetric-mode solver.

All 189 regression tests passed with real Elmer checks enabled in 114.88 s.
Sampled combined test-tree RSS peaked at 785.1 MiB, system RAM at 76.6%,
GPU memory at 1,378/8,192 MiB and GPU temperature at 49 C. CPU temperature
was unavailable. The committed guarded example then recovered all seven
groups in 5.53 s, with zero new solves, exactly equal saved arrays and exit
code 2 for the failed spatial screens. No full training ran.
