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

The proposal is explicitly blocked from wave launch by the integrator's
current 2,048 MiB workspace API ceiling. The finest vector/sample forecast
exceeds it, so the proposal requests 3,072 MiB. The preflight records
`integrator_api_supported=false` and `wave_launch_ready=false`. Implement
an explicitly bounded API extension, retain the float64 recurrence and
all free-memory/temperature guards, and verify CUDA equivalence against
both retained integrator source versions before launching. The current
single-version transition receipt cannot prove a mixed-version study;
extend its checking without dropping any per-version parity requirement.

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
