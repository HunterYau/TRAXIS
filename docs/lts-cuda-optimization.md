# LTS execution optimizations

PZT-study port: on `research/pzt-mesh-convergence`, this backend is also
available to the two-mesh scheduler. Run 007 selects graphs, fixed exact
fine/coarse steps 2.6702880859375/5.340576171875 ns, and explicitly disables
admission screening to preserve the user's study policy. The optional
`screen_stability` API flag defaults to true outside that study. The original
PZT-study solver/physics files match pre-port checkpoint `4a76879`; earlier
component resource-budget changes in that branch are retained. The details
below describe the originating LTS-only implementation and its fallback.

Branch `research/lts-cuda-optimization` starts from validated LTS checkpoint
`df32ef0e98f5e1ab0ed4576ea03b8f02a58e6490`. It excludes the subsequent PZT
mesh convergence study. The original solver and operator files are preserved.

The separate `traxis/gpu_lts_optimized.py` backend implements:

- Fixed FP64 state, operator and patch work buffers. Consistent PZT mass
  blocks are retained; gather-before-division preserves their exact inverse.
- Fusion of source subtraction/projection and the vector recurrence update.
  The plate stiffness, selected-element action and PZT matrix/reduction
  kernels use the original arithmetic. Fused vector kernels disable FMA
  contraction; roundoff differences still require parity verification.
- CUDA graph replay in chunks of at most 16 steps on a nonblocking stream.
  Source values and the sample index are device data, not constants frozen
  at capture. Initialization compiles kernels and resets all state before
  capture. Graph replay stops at the original energy and progress boundaries.

Energy measurements, finite-state checks, resource guards, the stability
screen and progress callbacks remain outside graph capture. Capture failure
is raised. The explicitly selectable original backend remains the fallback.
Ratios other than two dispatch to the original solver. Both fine-source
and coarse-source split-LFC forcing are supported; callables must be pure
functions of time because source values are sampled in advance.

Prepare a fresh original LTS benchmark using the existing CLI with:

```bash
python -m examples.pzt_local_time_study --prepare --root outputs/simulation-runs/lts-optimized-001 --execution-backend optimized
```

Use `optimized-no-graphs` to isolate buffer reuse and arithmetic fusion;
`original` is the default. Preparation records this choice in the plan,
and execution reads it from the plan. No old result is overwritten or reused
under a different backend. This change creates no PZT mesh-study schedule.

The API is otherwise the original `gpu_lts_simulate` signature, with optional
`use_graphs=True` and `graph_steps=16`. All precision, mesh and time-step
settings come from the caller. The previous PZT mesh study's disabled checks
and halved time-step changes are not included on this branch.

Small native-GPU comparisons cover partial fine regions, three resolved
PZT bodies with consistent mass and paired voltage probes, both source
sampling modes, long replay sequences, remainder chunks, non-divisible
diagnostic intervals, changing source pulses and original-ratio fallback.
They compare raw traces, derived sum/difference signals, envelopes and
modified energy. They do not establish full-plate speedup or temporal
accuracy. Test output and solver SHA-256 records are under
`outputs/cuda-optimization-audit` in this worktree.

Workspace estimates include the original diagnostic temporaries and fixed
source/patch buffers. CUDA graph metadata is runtime managed; CUDA allocation
failures still propagate. Only small local cases have been checked against
the 8 GB RTX 3060 Ti, not full-size plate capacity.

References: [CuPy stream capture](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html),
[CuPy graph replay](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Graph.html).
Synchronous device-to-host transfers are excluded from capture as required
by these APIs. No cloud pod is started or stopped by this implementation.
