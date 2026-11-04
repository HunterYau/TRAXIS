# Finite-plate waveform refinement

The local mechanical FEM does **not** yet produce spatially resolved research
waveforms. A noise-free intact-plate pilot completed its mesh/time sweep, but
the finest two meshes changed individual receiver traces by up to 146% in
relative L2 norm. The finest two time steps differed by at most 0.0209% on
the same mesh. Passing time integration checks does not repair spatial error.

## Reproduce the bounded check

Install the optional dependencies in requirements-meshing.txt, then run:

```powershell
.\.venv\Scripts\python.exe -m traxis plan-scenarios --config configs/waveform-control.json --output outputs/waveform-control-plan.json
.\.venv\Scripts\python.exe -m traxis check-waveforms --plan outputs/waveform-control-plan.json --settings configs/waveform-sweep.json --cache-dir outputs/waveform-refinement-groups --output outputs/waveform-refinement
.\.venv\Scripts\python.exe -m traxis plot-waveforms --sweep outputs/waveform-refinement --records 0 12 --output outputs/waveform-refinement-figures
```

Use fresh output directories for each attempt. A completed sweep that fails
the discrepancy screen returns exit code 2 and retains its report and traces;
an operational failure returns 1. Plotting a completed failed screen is
allowed. Each command defaults to 60 seconds and 512 MiB process RSS, with
additional system memory and GPU temperature/memory guards. Native meshing
cannot be preempted by the cooperative guard.

After confirming that an interrupted process has stopped, a new attempt may
use the same `--cache-dir`. Completed transmitter/frequency groups have
integrity checks and keys bound to the operators, source, probes, time step
and solver source. Recovery preserves the earlier failed report and never
silently overwrites corrupt groups. It still preflights and assembles meshes.

## Independent refinement axes

The plan fixes the specimen, material, aperture, pulse, output times, numerical
settings and discrepancy tolerances before propagation. Stochastic current
and control noise must be zero. Circular source/probe footprints use the
integrals described in [mechanical-patches.md](mechanical-patches.md), keeping
their physical area and center fixed under remeshing.

Three decreasing CAD mesh sizes are required. For each mesh the assembled
operator gives a conservative stable step bound. Their maximum required
substep count defines a common base N. All three spatial cases run on the
same finest time grid, with 4N substeps per saved sample in the default study.
On the finest spatial mesh, the temporal cases use N, 2N and 4N substeps.
The finest case is shared, giving five distinct cases. Each internal grid
lands exactly on the saved output times; no interpolation is used. A fixed
step that violates the stability bound fails before propagation.

Comparisons use float64 current-channel solver probes reconstructed from
checked unit-load cache groups with the declared gains and drift. Their
float32 versions must exactly reproduce the saved waveform bank. Traces
are compared on the entire saved window without fitted amplitude, phase,
time alignment or per-record normalization. Current/control traces are saved;
the present screen scores the current channel. A damaged study must also
establish baseline convergence before any downstream subtraction is trusted.

Each record retains raw relative L2 discrepancy, absolute RMS error, RMS
ratio error, cosine correlation, Hilbert-envelope discrepancy, envelope-peak
shift and carrier phase difference. Zero/weak reference traces have undefined
relative errors and cannot pass. Envelope peaks and whole-window carrier
phase are diagnostics, not identified A0/S0 arrivals. Reflections, overlapping
modes and rigid motion remain in these free-plate traces.

The provisional screens require defined records at both refinement pairs and
a maximum finest-pair relative L2 error below the predeclared tolerance: 5%
spatial and 1% temporal in this pilot. Three levels expose nonmonotone spatial
changes. The temporal change ratio gives an observed order when defined.
These are discrepancy screens, not a rigorous bound on continuum error or
measurement-based acceptance tolerances. Every result retains
`ready_for_training_data=false`, including a passing screen.

The wavelength table uses the independent Rayleigh-Lamb A0/S0 calculation at
each carrier and first-main-lobe upper zero, for stock and remaining recess
thicknesses. It reports wavelength divided by actual maximum element edge,
and stock thickness divided by maximum element z span. Unsupported dispersion
ranges are explicit. This table does not cover spectral tails or higher modes,
and a geometric ratio alone cannot establish bending accuracy.

## Measured pilot, 2026-11-04

The fixed control is a 40 x 40 x 3 mm intact free plate, eight 3 mm radius
mechanical patches, a 100 kHz two-cycle Hann burst, and 128 output samples
at 0.5 microseconds. It is much smaller than the intended 500 mm plate.
Nominal material properties and unit total force are assumptions; receivers
measure normal displacement in meters, not PZT voltage.

An initial Delaunay attempt hit the 60-second guard. Its middle mesh contained
a nearly coplanar tetrahedron with minSICN about 0.000233, forcing 80 base
substeps per output. Netgen optimization alone and the Frontal alternative
did not remove that sliver. HXT plus Netgen gave a minimum quality about
0.370 and eight required substeps on that mesh. HXT was selected from mesh
quality/stability-only probes, before examining waveform discrepancies; no
error tolerance, geometry, pulse or observation window was relaxed. All three
final mesh sizes used the same HXT/Netgen choice. Algorithms and before/after
quality measures are retained in provenance. The APIs and algorithm numbers
are documented in the [Gmsh manual](https://gmsh.info/doc/texinfo/).

| Check | Measured result |
| --- | --- |
| Requested mesh sizes | 3, 2, 1.5 mm |
| Minimum stable substeps by mesh | 6, 8, 10 per output sample |
| Spatial comparison time step | 12.5 ns on all meshes |
| Temporal comparison steps | 50, 25, 12.5 ns on the finest mesh |
| Coarse-to-middle maximum trace relative L2 change | 32.76% |
| Middle-to-fine maximum trace relative L2 change | 145.70% — failed |
| Coarse-to-middle temporal maximum change | 0.08351% |
| Middle-to-fine temporal maximum change | 0.02087% — passed |
| Observed temporal order, global difference norms | 2.00025 |
| Minimum fundamental/main-lobe wavelength/edge ratio | 2.94 |
| Stock thickness/maximum element z span | 1.0 |
| Solves / elapsed time | 35 grouped mechanical solves / 43.08 s |
| Sampled process RSS / system RAM | 122.9 MiB / 61.6% |
| Sampled GPU usage / temperature | 921 of 8,192 MiB / 44 C |

All solves used CPU. GPU readings are system observations, not allocations
attributed to this solver; CPU temperature was unavailable. The meshes still
contain elements spanning the entire thickness. Similarity of two coarse
results would have concealed the larger change at the third level.

The committed compact record is
[evidence/waveform-refinement-debug.json](evidence/waveform-refinement-debug.json).
Full per-record reports, meshes' diagnostic statistics and checked float64
traces remain under outputs/waveform-refinement-hxt. The PNG/SVG plots under
outputs/waveform-refinement-hxt-figures were visually checked for readable
axes, units and legends. They show preselected catalog records 0 and 12,
without new solves. Plotting checks plan/report fingerprints and raw file
hashes and rejects changed traces.

All 144 regression tests passed, including analytic gain/phase and delayed
packet checks, exact time halving, unstable-step rejection, interrupted-cache
recovery, independent mesh/time axes and changed-plot-input rejection. This
run sampled the monitor, test launcher and all observed interpreter descendants:
735.2 MiB combined RSS, 64.6% system RAM and 46 C GPU temperature at peak.
An earlier receipt measured the launcher incompletely and is superseded for
process-memory reporting. CPU temperature was unavailable.

This is software verification. Before research data generation, spatial resolution
must be demonstrated at affordable cost, followed by an independent elastic
reference, measured source/receiver coupling, supports/reflections and intact
physical agreement. PZT/adhesive/electrical properties remain required for
voltage simulation. No training updates were performed by this sweep.
