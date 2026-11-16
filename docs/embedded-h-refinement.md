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

The current fixed-case driver does not yet support these levels. Add support
only after the live time-refinement run and its independent CPU check finish,
preserve every old case unchanged and review the source transition. Reuse
the existing exact coarse 294,912-step controls only after validating their
source, CPU, physical-case and sample-grid bindings. Higher levels need new
paired runs and CPU checks. No full training or accepted bank is implied.

The analytical screens and protocol hash are retained in
`evidence/embedded-h-mesh-preflight-20261115.json`. The absence of a final
native operator/trajectory is explicit in that receipt.
