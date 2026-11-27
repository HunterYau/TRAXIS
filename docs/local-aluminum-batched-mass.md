# Separate batched mass solve

The stopped six-ring H100 run reached 880/35,714 steps. Its recent steady rate
was about 5.57 coarse steps/s (approximately 104 minutes remaining at that point).
Those logs measure the complete step, not the individual operator costs.

The reference local-aluminum adapter performs five sequential independent mass
block solves per inverse: two constrained aluminum boundary blocks and three
PZT/host consistent mass blocks. Each has three Cartesian RHS. Each LTS coarse
step applies the mass inverse at least twice, with extra applications for the
original energy diagnostics. The reference also allocates gather temporaries
and copies full aluminum vectors during every host action. These are confirmed
implementation costs; their share of H100 runtime has not been profiled.

`traxis/gpu_local_aluminum_batched.py` retains the reference GPU implementation
and all its matrices. Its separate operator concatenates the already computed
exact sparse LU factors into a block diagonal system, including their original
row and column permutations. One persistent native cuSPARSE solve then exposes
the independent blocks to GPU scheduling concurrently. It gathers/scatters all
three Cartesian RHS with persistent buffers. There is no dense inverse, mass
lumping, iterative approximation, or change to the LTS recurrence. Cached native
analysis remains outside CUDA Graph capture. The reference inverse remains
available as `system.reference_inverse_mass_inplace`.

Small RTX 3060 Ti checks:

| Check | Result |
|---|---:|
| Direct-sum CPU factor permutation error | 1.45e-16 relative L2 |
| Synthetic five-block GPU mass inverse vs sequential | 0 relative L2 |
| Synthetic mass graph replay vs sequential | 0 relative L2 |
| Sequential / batched mass graph replay | 3.41 / 1.25 ms |
| Mass-operation speedup in this synthetic case | 2.74x |
| Small bonded model, stiffness and inverse mass | 1.45e-15 relative L2 |
| Small bonded model, graph vs ordinary trace | 0 relative L2 |
| Small bonded model, unchanged CPU recurrence vs GPU | 4.91e-16 relative L2 |

The synthetic mass test has the same five-block architecture but different
matrix entries and sizes from production. Its 2.74x speedup does not establish
the H100 full-run speedup. For illustration, if mass solves consume 50%, 75% or
90% of runtime, a 2.74x mass acceleration would imply about 1.47x, 1.91x or 2.34x
overall acceleration. The actual share and production speed must be measured.

The optimized plan is separate at
`outputs/simulation-runs/spectral-q2-local-aluminum-batched-001/plan.json`.
It selects `local_execution.mass_solve_backend = "batched"`. It retains six
rings at 4.9/9.8 ns, eight rings at 4.7/9.4 ns, the same host/PZT mesh, FP64,
physical inputs and six-frequency reconstruction. The old archive and GPU
adapter remain references. The batch adds another sparse factor payload to
retain that fallback; report the exact device payload at build time and check
free GPU memory before allocating. Neither a new full simulation nor a pod
start was performed during this work.
