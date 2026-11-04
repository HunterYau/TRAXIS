# Independent quadratic guided-wave reference

Quadratic tetrahedra offer a promising way to reduce the bending-resolution
cost observed in the local linear-tet solver. A bounded Elmer P2 dispersion
study reached a maximum finest-grid frequency error of 0.0677% against the
independent Rayleigh-Lamb solution. Its overall predeclared screen **failed**:
one small S0 thickness-refinement error increased instead of strictly
decreasing. The failed result is retained. No research waveform bank or
training readiness follows from this study.

## Commands and scope

Use the existing portable/installed solver described in
[elmer-reference.md](elmer-reference.md):

```powershell
.\.venv\Scripts\python.exe -m traxis verify-elmer-waves --elmer-home outputs/reference-tools/elmer-26.1/ElmerFEM-nogui-nompi-Windows-AMD64 --output outputs/elmer-p2-guided-reference
.\.venv\Scripts\python.exe -m traxis plot-elmer-waves --sweep outputs/elmer-p2-guided-reference --output outputs/elmer-p2-guided-figures
```

The verifier returns 2 on a completed failed numerical screen, 1 on an
operational error, and 0 on a passing screen. Plotting a completed failed
screen is allowed and launches no solver. Each output needs a new or empty
directory. Defaults are 60 seconds and 512 MiB combined monitor/external-tree
RSS, with system memory and best-effort GPU/temperature guards. CPU execution
uses one thread per external process. No global installation changes occur.

The default targets are 100/200 kHz in uniform 1/3 mm isotropic stock. The
200 kHz point is the upper first-main-lobe zero of the two-cycle, 100 kHz
Hann pulse used in the earlier finite-plate pilot. It does not cover the full
pulse spectrum, all research frequencies, every remaining thickness, recess
scattering, damping or a DAQ filter. These frequencies/thicknesses can be
explicitly supplied through `--frequencies-khz` and `--thicknesses-mm`, with
at most eight combinations. Unsupported traction-root ranges fail before
any external assembly.

This is an infinite-plate benchmark. Lateral periodic faces remove physical
edges only for comparison with the analytic dispersion relation. Top/bottom
surfaces stay traction-free. There are no finite-plate arrivals, reflections,
supports, source/probe patches, PZT or voltage readouts in this check.

## What is computed

An independent traction-root calculation gives the A0 or S0 wavenumber at
each target frequency. That wavenumber sets a small periodic cell's width
and Bloch phase. Elmer assembles elastic stiffness and consistent mass from
geometry/material inputs; the generalized eigenproblem then supplies the
numerical frequencies. The target frequency is not a stiffness/mass input.

ElmerGrid raises a six-tet structured cell to 10-node quadratic tetrahedra.
Native node/type/connectivity checks require complete P2 nodes and bound
the problem before assembly. The finest cell has 81 nodes and 243 mechanical
degrees of freedom. A complex projection ties opposite x faces with
`exp(i*k*Lx)` and y faces with phase one. It suppresses y displacement to
select the sagittal sector and retains independent interior x/y P2 values.
Applying a prescribed exponential to interior values would constrain the
approximation being tested, so only corresponding boundary nodes are tied.

The projection uses a conjugate transpose to condense the independent Elmer
operators. Hermitian checks precede the positive generalized eigen solve.
The lowest two sagittal branches supply A0/S0 at these low-frequency targets.
The benchmark does not claim higher-mode identification beyond this range.
The independent implementation and formula references are linked from
[elmer-reference.md](elmer-reference.md) and the Rayleigh-Lamb notes in
[implementation-status.md](implementation-status.md).

Each target requires five operator exports. In-plane cell counts 4/8/16 per
wavelength share four thickness cells. Thickness counts 1/2/4 share sixteen
in-plane cells per wavelength. The finest case is shared by both axes.
These are structured cell counts; each cell contains several tetrahedra and
quadratic interpolation points. Refining thickness also changes the 3D
tetrahedral triangulation; this is not a nested one-dimensional basis study.

The plan freezes the target grid, both refinement axes, 0.5% finest-frequency
error tolerance and strict error-decrease requirement before any assembly.
The overall screen requires both axes to strictly decrease at every target
and every finest error to stay below 0.5%. Matching analytic frequencies
within tolerance alone does not override a failed decrease requirement.
Reports retain source/executable and all mesh/SIF/matrix/log hashes. Figures
check the report and plan fingerprints and retain their hashes in a manifest.

## Measured results, 2026-11-04

The 40-export study completed in 21.58 seconds. Finest-grid errors were:

| Target frequency | Uniform thickness | A0 relative error (%) | S0 relative error (%) |
| --- | --- | --- | --- |
| 100 kHz | 1 mm | 0.06289 | 0.001843 |
| 100 kHz | 3 mm | 0.05259 | 0.002178 |
| 200 kHz | 1 mm | 0.05279 | 0.001982 |
| 200 kHz | 3 mm | 0.06764 | 0.002986 |

All in-plane sequences and seven of eight thickness sequences strictly
decreased. For 100 kHz S0 in 1 mm stock, the thickness errors were
0.001735%, 0.001817% and 0.001843%. This small nonmonotone component has not
been isolated; the predeclared criterion was not relaxed after observing it.
Both useful absolute accuracy and the failed refinement diagnostic are
reported. The plots show the nearly flat S0 thickness sequence alongside
the larger A0 refinement changes.

A separate nine-case follow-up fixed the problematic target and tested
16/32/64 in-plane cells at each of 1/2/4 thickness cells. Its own plan was
written before those solves, with no replacement pass criterion. At four
thickness cells, the S0 errors fell from 0.001843% to 0.0001242% and then
0.000009380%, giving observed in-plane orders 3.89 and 3.73. Both the 32- and
64-cell thickness sequences decreased. This supports an in-plane dispersion
interaction behind the earlier plateau; it does not rigorously isolate every
error component. The follow-up took 4.58 seconds and preserves the original
failed screen. Its full inputs/matrices and report remain under
outputs/elmer-s0-floor-diagnostic, with compact results in the same evidence.

Sampled combined RSS reached 109.9 MiB, system RAM 61.7%, GPU memory
921/8,192 MiB and GPU temperature 47 C. CPU temperature was unavailable.
GPU values describe the system; these calculations used CPU. The PNG/SVG
figures were rendered and visually checked for readable axes, units, legends
and the failed-screen title. Evidence is in
[evidence/elmer-waveguide-debug.json](evidence/elmer-waveguide-debug.json).

All 156 regression tests passed with both real Elmer integration tests
enabled. The full suite took 60.25 seconds, reaching 779.7 MiB sampled
monitor/test-tree RSS, 64.9% system RAM and 47 C GPU temperature. CPU
temperature was unavailable. Tests include periodic face phases, independent
interior values, conjugate reduction, moved/unmatched nodes and actual
quadratic-mode agreement with the independent traction root.

The earlier linear-tet Bloch benchmark used lumped mass and different meshes;
these results are not a controlled element-order-only cost comparison.
They support further high-order reference work, while the finite-plate local
waveform mesh screen remains failed. Finite-boundary/recess/source/probe
convergence, transient time refinement, physical supports and actual PZT/
adhesive/electrical parameters are still required for research data.
