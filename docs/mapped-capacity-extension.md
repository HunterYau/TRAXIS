# Explicit mapped-GPU capacity for the finer-cell study

The finest declared Q4 mesh has 12.43-12.57 million vector coefficients.
Review found a second 12-million ceiling in the mapped GPU preflight, beyond
the mesh generator's separate cap. The isolated extension adds an explicit
`max_vector_dofs` argument to the GPU screen and builder, bounded at 24 million.
The default remains 12 million. Only the frozen h-refinement cases request
the larger value; ordinary controls keep their original call settings.

The host/device byte forecasts, element limit, float64 arithmetic, free-device
headroom, positive mass/geometry checks and existing resource/thermal guards
are retained. The CUDA kernel bytes match the original baseline exactly.
This is a capacity policy change, not a precision or discretization reduction.

Nine CPU tests pass. A broadcast count-only fixture checks opt-in above the
default and confirms the larger cap cannot bypass the memory limits. It is
not a physical mesh or forward-accuracy result. Existing mapped preflight and
fourteen legacy-case regressions also pass in that run.

A separately guarded small Q4 CUDA action test passes: default versus explicit
capacity differs by less than 1e-12 relative, independent CPU action by less
than 1e-11, and mass agrees exactly. The test asserts these bounds; it does not
claim a bitwise-identical atomic reduction or validate a large h trajectory.
Bindings are in `evidence/mapped-capacity-extension-debug-20261115.json`.

The main fine temporal run remains frozen. Merge this extension after that
run and its independent CPU check finish, regenerate the source-transition
certificate for the new review hash and verify actual larger-grid capacity
and stability before propagating. Never remove allocation limits to force
the larger case through. Full finer-grid numerical acceptance remains open.

## Actual finest recessed CPU preflight

An actual 12,431,655-DOF mesh and mapped CPU operator now complete this
preflight in 261.477 seconds. All GLL Jacobians and masses pass their existing
positive/finite checks; minimum Jacobian is 1.17773e-10 and maximum geometry
condition is 3.58005. The total quadrature-volume discrepancy from the analytic
recessed solid is 3.64e-14 relative. This is a geometric diagnostic, not a
scattering-accuracy estimate.

The selected step is 1.695421 ns, versus a conservative stability bound of
3.257927 ns: 52.04% of that bound, below the retained 80% ceiling. GPU storage
forecast is 2,091.30 MiB and host forecast 1,558.51 MiB. Actual CPU RSS peaks
at 1,043.77 MiB, system RAM at 61.1%, with at least 6,316.66 MiB available.
The main fine temporal calculation was running concurrently on the GPU.

No GPU operator or trajectory was allocated. The complete source snapshot,
case, package identities, operator-array hashes and summary are retained;
the full CPU operator is not stored. Source/runtime identities are rechecked
after assembly. See `evidence/embedded-h-finest-cpu-preflight-20261115.json`.
The preflight command is `python -m examples.embedded_h_preflight --h-level 2
--recess --output outputs/fresh-preflight`; use a fresh output directory.
