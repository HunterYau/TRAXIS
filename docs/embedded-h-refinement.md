# Additional spatial refinement after the scattering failure

The original Q3/Q4/Q5 study retains its failed scattering result at receiver
3: 1.344761% finest change against the predeclared 1% threshold. A further
study will refine cell sizes at fixed Q4, retaining that threshold. Its
protocol is `configs/embedded-h-protocol.json`; no new wave outcome has been
inspected when declaring it.

All three levels use 294,912 steps over 500 microseconds, cubic geometry,
the same 3.175 mm plate and recess, 100 kHz two-cycle source and physical
load/receiver disks. Exterior spacing is 5 / 3.75 / 3 mm, angular subdivisions
12 / 16 / 20 and radial counts (2,6) / (3,8) / (4,10). Each receiver's intact,
recessed and recessed-minus-intact traces must have identifiable, decreasing
differences and a finest relative change below 1%. No amplitude fit, phase
alignment or interpolation is permitted. This is a new study rather than a
replacement verdict for the original p-refinement.

Analytical mesh screens predict intact vector DOF counts of 4,507,515,
8,090,091 and 12,567,987; recessed counts are 4,462,575, 8,005,599 and
12,431,655. Every mesh forecast remains under 512 MiB. The finest mesh needs
an explicit DOF cap above the older 12-million default; its proposed bound
is 24 million. This does not establish total host/device feasibility: actual
operator, workspace, stability, RAM and GPU checks remain mandatory before
propagation. Resource ceilings and float64 accuracy must be preserved.

The original fixed-case driver did not support these levels. Support was
developed separately, then integrated after the coarse time run and its CPU
check finished, preserving every old case and reviewing the source transition. Reuse
the existing exact coarse 294,912-step controls only after validating their
source, CPU, physical-case and sample-grid bindings. Higher levels need new
paired runs and CPU checks. No full training or accepted bank is implied.

The analytical screens and protocol hash are retained in
`evidence/embedded-h-mesh-preflight-20261115.json`. The absence of a final
native operator/trajectory is explicit in that receipt.

## Isolated control implementation

The `research/embedded-h-refinement` branch adds these frozen cases through
`embedded_wave_budget --h-level 0`, `1` or `2`. Each uses Q4 and 294,912
steps; add `--recess` for material loss. The general wave CLI rejects this
scope so the explicit 7,200-second / 4,096-MiB execution profile is required.
System RAM remains limited to 89.9%, GPU memory to 90% and GPU temperature
to 80 C. Actual operator/workspace and stability checks still precede solving.

Twenty-four selected tests pass without skips. A baseline fixture captured
before editing verifies the exact case, mesh-screen and extended-budget
identities of all fourteen legacy controls. A bounded archived/current CUDA
check passes with every dense/sparse trajectory, energy and final-state
difference zero. This verifies legacy behavior; finer-grid operator and full
trajectory checks have not run. The paired h-study assessor and larger-case
CPU wrapper remain to implement before any acceptance claim.

Development was isolated in `outputs/embedded-h-development` while the main
checkout's coarse time-refinement run and CPU check were pending. The main frozen
source hashes were explicitly rechecked unchanged. The portable interpreter
normally imports the main checkout: isolated unit tests use an ignored runtime
with the development path configured. The copied interpreter could not locate
CUDA headers/NVRTC; CUDA parity instead used the original configured
interpreter with an explicit development import path. Failed attempts are
retained. No global environment or installed packages were changed.

Bindings and limits are in `evidence/embedded-h-controls-debug-20261115.json`.
The controls are now integrated on the main research branch. Twenty-four
selected tests pass again there, and CUDA transition v7 retains exact parity.
The fine temporal run is active; its source manifest remains frozen. Actual
finer-grid trajectories and their paired assessment remain pending.
