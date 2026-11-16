# Coupled spatial refinement proposal

The existing in-plane study keeps thickness layers fixed. Even a passing
in-plane result will leave full three-dimensional convergence unproven.
The separate `configs/embedded-xyz-proposal.json` therefore refines both
in-plane cells and the lower/upper thickness layers, at fixed Q4:

| Level | Angular | Radial | Exterior spacing | Thickness layers |
| --- | --- | --- | --- | --- |
| 0 | 12 | (2, 6) | 5 mm | (1, 1) |
| 1 | 14 | (3, 7) | 4.5 mm | (2, 2) |
| 2 | 16 | (4, 8) | 4 mm | (3, 3) |

The physical plate, recess, source, receivers and acceptance thresholds
remain the same as the mechanical controls. The proposal uses 491,520
steps over 500 microseconds and common integer sample selection. The
same 1% finest-change limit applies to every intact, recessed and
defect-induced receiver trace, with decreasing identifiable differences,
independent CPU checks and no amplitude fitting, phase alignment or
interpolation. No wave data have yet been generated for this proposal.

Both actual finest CPU meshes/operators were constructed successfully:

| State | Vector DOFs | Stability bound | Used fraction | GPU forecast | Peak RSS |
| --- | --- | --- | --- | --- | --- |
| Intact | 20,352,675 | 3.086625 ns | 32.96% | 3,499.66 MiB | 1,697.26 MiB |
| Recessed | 20,062,335 | 3.208633 ns | 31.70% | 3,448.84 MiB | 1,670.84 MiB |

The proposed step is 1.017253 ns, below the unchanged 80% stability margin
in both actual operators. Geometry, positive mass, volume and retained
array hashes were checked. Setup took 186.871 and 190.589 seconds; system
RAM peaked at 69.1% and 68.9%. CPU temperature could not be verified from
a supported sensor. GPU figures are allocation forecasts; no GPU operator
was allocated and no wave propagation or GPU throughput was tested.

The first preflights exposed the integrator's 2,048 MiB workspace API
ceiling before any GPU launch. The finest vector/sample forecast exceeds
it, so the proposal requests 3,072 MiB. The explicit API now permits up to
4,096 MiB, with its 512 MiB default, float64 recurrence, storage forecasts,
free-VRAM headroom and device/resource guards unchanged. A repeated actual
finest recessed CPU preflight preserves every prior array hash and now
records `integrator_api_supported=true`, with `wave_launch_ready=false`.

Both retained integrator versions were separately compared with the new
source using actual CUDA integration: dense/sparse traces, energies and
final states match exactly from zero and nonzero initial conditions. A
bound multi-version receipt requires every archived version needed by a
study and checks every supplied child against the original requirements.
Missing, duplicate, altered, escaped, incomplete and failed proofs reject;
an invalid extra version cannot hide behind a valid subset.

All 32 focused checks passed without skips, including genuine small-Q4 GPU
launches using the planned H and XYZ step/stride/workspace interfaces.
These launch tests intentionally stop at synchronized batch 16. They do
not establish a complete window or an actual 20-million-DOF GPU allocation.
The paired coupled-XYZ wave launcher, assessor and complete trajectories
still need implementation and validation. See
`evidence/explicit-workspace-extension-20261115.json` for the bound checks.

The preflight can be reproduced from this branch in a standard project
environment with `python -m examples.embedded_xyz_preflight --level 2
--recess --output outputs/xyz-recess-fresh` and the corresponding intact
command without `--recess`. Each output directory must be fresh. The
current host's portable runtime requires the documented checkout-path
bootstrap when running this isolated branch; CUDA headers are available
through the original root interpreter, not the copied CPU interpreter.

Actual finest proofs and all-level analytic counts are bound in
`evidence/embedded-xyz-finest-cpu-preflight-20261115.json`. Coarse/middle
counts are forecasts, not completed constructions. This work does not
replace the preserved 1.344761% spatial-scattering failure or establish
full 3D convergence, calibrated PZT voltages, real-specimen similarity,
an accepted training bank or completed model training.
