# Native cuSPARSE graphs for the matrix-free PZT backend

`traxis/gpu_native_cusparse.py` adds an opt-in ctypes bridge to the existing
CUDA-12 cuSPARSE library. `traxis/gpu_spectral_native_graph.py` adapts the separate
matrix-free PZT operator to the existing graph executor. The prior cached CuPy
solver, sparse operator, precomputed operator and all LTS files are retained.

## ABI and library compatibility

The native signature was checked against NVIDIA CUDA 12.9 documentation and
CuPy's Cython wrapper:

```c
cusparseStatus_t cusparseSpSM_solve(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void* alpha, cusparseConstSpMatDescr_t matA,
    cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC,
    cudaDataType computeType, cusparseSpSMAlg_t alg,
    cusparseSpSMDescr_t spsmDescr);
```

TEN arguments, no workspace argument. The workspace is attached during analysis
and retained by the plan. The bridge declares every ctypes argument and return
type explicitly; enum/status values are C ints and pointers are 64-bit.

- Windows resolves the already-loaded `cusparse64_12.dll` with GetModuleHandleW
  and verifies the ctypes loader returns the same module handle.
- Linux attaches to the single loaded `libcusparse.so.12` found in the process
  maps with RTLD_NOLOAD; it does not search for another toolkit version.
- The bridge rejects an unexpected CUDA runtime major, verifies its native
  library version equals CuPy's, and uses dedicated handles with host pointer
  mode. Local results: runtime 12090, cuSPARSE 12510 through both interfaces.
- CUDA-13 driver compatibility does not imply loading a CUDA-13 runtime/library.
  This implementation specifically targets the project's pinned CUDA-12 stack.

Sources: [NVIDIA SpSM API](https://docs.nvidia.com/cuda/archive/12.9.1/cusparse/index.html#cusparsespsm),
[CuPy v14.2.0 bindings](https://github.com/cupy/cupy/blob/v14.2.0/cupy_backends/cuda/libs/cusparse.pyx).

## Ownership, streams and fallback

Each `NativeCachedLU` wraps the original `CachedLU`, sharing its exact FP64
factors, permutations, RHS, intermediate and result arrays. The original CuPy
plans remain alive as a selectable fallback. Native lower/upper plans own their
SpSM descriptors and analysis workspaces and retain all borrowed descriptors.
Analysis is performed once, before capture. During capture, only the native solve
and fixed-buffer CUDA kernels run; CuPy's rejecting `_setStream` wrapper is not
entered.

Each LU has a dedicated persistent cuSPARSE handle. Bind it to the current stream
before capture; mismatched streams raise an error. Stream changes and close wait
for that handle's previous stream. Retain the system, graph, factors, descriptors
and buffers until replay has ended; close the system after the final replay.
Concurrent execution on independent streams requires separate system instances.

`couple_native(base, attachments, masses, allow_fallback=False)` requires native
support by default. An explicit `allow_fallback=True` returns the original cached
matrix-free system if native initialization fails and marks graph support false.
`system.set_native_enabled(False)` selects retained CuPy plans outside capture;
use the original ordinary executor in that mode. No silent fallback occurs
during capture or graph execution.

## Unchanged integration

The existing optimized executor assumes a hook named `dense`; the new action
class overrides that hook to evaluate implicit matrix-free PZT actions and copy
into the executor's reusable output buffers. It allocates no dense patch matrix.

The adapter creates isolated function-global dictionaries with only the operator
factory changed. The `_execute` and simulation validation functions retain the
EXACT original Python code objects, defaults, validation and LTS recurrence.
The original module globals are not monkeypatched. Tests assert code-object
identity, in addition to Git diff and SHA-256 checks of source files.

The adapter currently supports the existing p=2 explicitly unchecked study policy
(`screen_stability=False` must be supplied). It does not silently disable a
requested screen or substitute an incompatible dense mass-normalization screen.
No scheduler or frozen study plan is switched automatically. A separate full
study preparation/validation is still needed before deployment.

## Small local verification

```text
python -m examples.spectral_native_graph_check --output outputs/spectral-native-graph-check-001
```

On Windows, add `--local-cuda-root <original-project-root>` for the existing
isolated CUDA DLL directories. The check does not install anything or contact
RunPod. It reserves 512 MiB of free GPU memory before starting small tests;
local precheck found over 6 GiB free GPU memory and about 4.5 GiB available RAM.

Verified on RTX 3060 Ti / CUDA runtime 12.9:

- Correct native signature/library instance/version.
- Native and cached fallback LU solves versus SciPy, with nontrivial permutations,
  changing inputs, and one/three RHS columns.
- Successful native graph capture and repeated replay with changing RHS buffers.
- Matrix-free full/local stiffness, consistent mass and inverse-mass actions.
- Tiny three-PZT model: 1,431 DOFs, 65 unchanged coarse-source LTS steps;
  four derived voltage channels, envelopes, energy and progress checkpoints.
- Additional 33-step fine-source graph parity, awkward graph chunk sizes and
  energy sampling boundaries.
- Explicit fallback after injected native initialization failure.
- Eight native analysis calls in the three-PZT system (four factors, two plans
  each), unchanged through replay. The reference CuPy plans remain separate.
- CuPy retained pool bytes stayed at 2,975,744 across ten further replays.
  This checks CuPy allocation growth, not every internal NVIDIA allocation.

Largest relative numerical difference in the final run: **3.56e-16**.
Waveform/energy agreement is component-level evidence, not full-plate convergence.

| Final warmed full/local action-pair measurement | Ordinary GPU | Graph replay |
| --- | ---: | ---: |
| Median host wall time | 3.290 ms | 2.957 ms |
| Median CUDA-event time | 3.286 ms | 2.955 ms |

Three batches of ten warmed identical action pairs, graph construction excluded:
**1.11x speedup**, about 10% less elapsed time. An earlier small run showed a
larger gain; host/system timing varies. These numbers do not predict a full-plate
or H100 speedup. The 65-step executor measurement including graph construction
was 0.241 s ordinary and 0.253 s with graphs: capture cost outweighs replay
savings in that very short run with seven graph variants.

Results: `outputs/spectral-native-graph-check-001/gpu-check.json`.
Protected before/after hashes and solver Git diff checks are stored alongside it.
No full simulations were started and no RunPod processes were touched. The Linux
loader path is implemented but has not been executed on RunPod during this work.
