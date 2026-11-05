# Quadratic finite-plate mesh and time refinement

`check-reference-waveforms` measures raw receiver changes for one P2
transmitter/frequency group, using independent spatial and temporal axes.
It builds on the [checked sparse mechanical reference](elmer-waveforms.md).
No training or automatic data promotion occurs, including when a discrepancy
screen passes.

```powershell
.\.venv\Scripts\python.exe -m traxis check-reference-waveforms --plan outputs/waveform-control-plan.json --settings configs/elmer-plate-refinement.json --elmer-home outputs/reference-tools/elmer-26.1/ElmerFEM-nogui-nompi-Windows-AMD64 --cache-dir outputs/elmer-p2-refinement-groups --output outputs/elmer-p2-plate-refinement --max-seconds 360 --max-rss-mib 768
.\.venv\Scripts\python.exe -m traxis plot-reference-waveforms --sweep outputs/elmer-p2-plate-refinement --records 0 3 --output outputs/elmer-p2-plate-refinement-figures
```

The control plan is created with `plan-scenarios` and
`configs/waveform-control.json`, as described in elmer-waveforms.md. Use
fresh output directories. The five-case study uses 3, 2 and 1.5 mm target
mesh sizes with 16 internal steps per output sample. The finest mesh also
runs at 4 and 8 substeps, sharing its 16-substep case with the spatial axis.
For the 0.5 microsecond output grid, these are 125, 62.5 and 31.25 ns.
All cases retain the same physical plate, source, apertures and output times.

The plan fixes the scenario, state, actions, numerical settings, source
hashes and thresholds before meshing. Every case retains its source snapshot,
kernel/runtime hashes, native operators or checked-cache identity, worker
receipt and float64 traces. Source/kernel/runtime/scenario/action consistency
is checked across the study. Temporal cases must have identical native mesh
and aperture file hashes. Saved traces and times must exactly reproduce the
values used for the comparisons.

The report uses the same raw-trace discrepancy definitions as the
[P1 study](waveform-validation.md): relative L2 and absolute RMS errors,
amplitude ratio, correlation, envelope and carrier phase diagnostics. It
does not fit amplitude, align time/phase or interpolate. Both adjacent pairs
must have resolved reference signals; the maximum finest-pair relative L2
change must be at most 5% for space and 1% for time in this control plan.
The global temporal difference ratio gives the observed order. These are
provisional discrepancy screens, not bounds on continuum error. The main-lobe
Rayleigh-Lamb wavelength/actual-edge table remains a diagnostic.

Completed failed screens return 2. Operational failures return 1 and preserve
the partial report, per-case results and native logs. After confirming the
process has stopped, repeat into a fresh output directory with the same
checked cache to recover completed groups. Incomplete groups are recomputed;
thresholds and failed attempts are preserved. The total solve budget is five
distinct study cases per attempt; interrupted work also consumes time and
must be included when reporting the study's actual cost.

The example uses a 768 MiB combined monitor/descendant RSS cap for the finer
P2 factorizations, with a 360-second bound per attempt. Other system memory,
available RAM and GPU temperature/memory limits remain active. Computation
uses CPU; CPU temperature is unavailable on this host. Native meshing is
bounded by preflight/count checks but is not preempted by the cooperative
guard. The external assembly and sparse worker are monitored and stoppable.

Separate current/intact convergence is required before interpreting a defect
residual. Passing this one intact transmitter/frequency group would still
leave other paths/frequencies, recess scattering, the 500 mm plate and bonded
PZT/adhesive/electrical/physical validation open.

## Measured control, 2026-11-04

The completed study used the same 40 x 40 x 3 mm intact free plate as the
P1 pilot, with eight radius-3 mm apertures, TX 0 and seven receivers at
100 kHz. The two-cycle Hann source, material and 128 output samples at
0.5 microseconds were unchanged. These are equivalent mechanical loads and
displacement probes, not bonded transducer voltages.

| Mesh target | P2 nodes / degrees of freedom | Tetrahedra | Finest-step case time |
| --- | --- | --- | --- |
| 3 mm | 3,777 / 11,331 | 1,928 | 21.04 s |
| 2 mm | 7,289 / 21,867 | 3,735 | 45.43 s |
| 1.5 mm | 15,478 / 46,434 | 8,543 | 129.90 s |

| Predeclared check | Measured raw maximum relative L2 change | Result |
| --- | --- | --- |
| 3 to 2 mm, dt=31.25 ns | 12.27% | Diagnostic pair |
| 2 to 1.5 mm, dt=31.25 ns | 30.84% | **Failed 5% spatial screen** |
| 125 to 62.5 ns, finest mesh | 0.8825% | Diagnostic pair |
| 62.5 to 31.25 ns, finest mesh | 0.2205% | Passed 1% temporal screen |

The observed temporal order was 1.99917. All seven reference signals were
resolved. The spatial changes increased across the three levels, so even
the small-plate P2 traces remain unsuitable for research training. A passing
time screen and matching independent operators do not establish spatial
accuracy. The finest mesh still contains elements spanning the full 3 mm
thickness; its maximum edge is 3.382 mm and minimum diagnostic main-lobe
wavelength/edge ratio is 2.94. These observations motivate separate
through-thickness/in-plane checks, without establishing the cause of the
waveform discrepancy.

The five solves completed in 327.70 seconds, with sampled combined RSS
640.4 MiB and system RAM at 79.4% maximum. Available memory stayed above
3,349 MiB. System GPU observations peaked at 1,690/8,192 MiB and 51 C;
the simulation allocated no GPU arrays. CPU temperature was unavailable.
Maximum linear residual was 1.01e-15 and postburst relative energy range
1.57e-14 across the five cases. These algebraic checks do not alter the
failed spatial result. The preselected TX 0 to RX 1/RX 4 plots were exported
as PNG/SVG and visually checked, including raw spatial and time differences.

The initial attempt reached its 120-second limit after two completed cases.
An unchanged-physics recovery reused both groups but hit the 768 MiB RSS
guard on the finest system. Isolated factorization probes then identified
off-diagonal pivot fill and explicit L/U diagnostic copies as resource costs.
Symmetric-mode ordering plus the stored factor count completed the same
fine matrix at 538.1 MiB sampled combined RSS and 1.20e-15 solve residual.
The ODE, energy and native-transient tests passed before rerunning all five
cases with the new source and unchanged thresholds. The final workflow
remained under the original memory cap. Failed attempts, costs and source
versions are retained in
[evidence/elmer-plate-refinement-debug.json](evidence/elmer-plate-refinement-debug.json).
The completed and interrupted study attempts together consumed 481.48 s;
the separate factor probes and regression/cache checks have their own receipts.

All 183 regression tests passed with real Elmer checks enabled in 112.02 s.
Sampled monitor/test-tree RSS peaked at 782.6 MiB, system RAM at 78.3%, GPU
memory at 1,531/8,192 MiB and GPU temperature at 48 C. CPU temperature was
unavailable. A separate native CLI recovery check reused all five groups
with zero new solves, exactly reproduced every saved time/displacement
array, and retained exit code 2 for the failed spatial screen. It took 7.49 s.
