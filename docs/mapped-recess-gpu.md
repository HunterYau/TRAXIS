# Mapped recess CUDA operator

The current CUDA path applies float64 elastic stiffness on the explicit
curved recess connectivity at the revised 3.175 mm stock thickness. This
checkpoint covers bounded 40 by 30 mm controls, not full-size wave accuracy.
Historical 3 mm reference evidence remains unchanged.

`traxis.gpu_mapped_tensor.build_gpu_mapped` accepts a checked CPU
`MappedTensorSystem`. One CUDA block gathers an element's displacement,
evaluates physical gradients and stress using its per-GLL-point inverse
Jacobian, applies the weak transpose derivative and scatters shared-node
forces with float64 atomics. It stores geometry and connectivity, without
dense element stiffness matrices or a global sparse matrix.

The GPU system retains the CPU positive GLL mass and its conservative
diagonal-energy stability bound. The existing device-resident Verlet
integrator can use this operator directly. Memory forecasts include explicit
connectivity, variable geometry, sixteen global vectors and kernel storage;
sample/probe storage is checked separately by the integrator. Counts and
memory limits reject before CUDA import. Construction also checks available
VRAM, shared memory and support for float64 atomics.

## Fresh native comparison

Eight cases pass with source/runtime snapshots and monitored workers. Geometry
degree is three; the selected circular wall uses four elements per quadrant.
Native deformed-H1 element matrices, nodal basis transforms, positive mass,
global applications, six rigid motions and affine strain are checked afresh
at 3.175 mm. The earlier CPU reference's 3 mm results are not relabeled.

| Geometry | Degree | Vector DOFs | Elements | Conservative stable step, ns |
| --- | --- | ---: | ---: | ---: |
| Coarse recess | Q3 | 1,488 | 13 | 23.23488 |
| Coarse intact | Q3 | 1,848 | 18 | 21.59124 |
| Selected recess | Q3 | 7,212 | 64 | 13.32689 |
| Selected intact | Q3 | 9,597 | 96 | 12.99338 |
| Selected recess | Q4 | 15,855 | 64 | 7.59152 |
| Selected intact | Q4 | 21,627 | 96 | 7.40806 |
| Selected recess | Q5 | 29,538 | 64 | 4.87865 |
| Selected intact | Q5 | 40,953 | 96 | 4.76368 |

These are stability bounds for each represented discrete system, not selected
recording intervals or evidence of wave resolution. The native comparison
checks 511 elements and 200,352 local columns. Both small Q3 global matrices
also compare all 3,336 GPU columns with native stiffness. Maximum independently
recomputed column error is 4.07e-14. For the selected meshes, the GPU action
agrees with independent scatter from retained native element matrices within
3.32e-13 relative; native-dual field comparisons are within 1.28e-14.

The coarse Q3 intact/recess cases use their highest native eigenmode in free
and analytic cosine-driven checks. At 32/64/128 steps, displacement and
velocity error ratios range from 4.00019 to 4.00111. Unforced modified-energy
relative range is below 9.82e-16. These intervals are only 54.98/57.43 ns and
use synthetic modal loads. They do not verify physical PZT pulses or a
complete plate-wave recording window.

## Verification and resources

The final focused regression passes 18 tests without skips in 9.531 test
seconds (10.765 monitored seconds). It includes Q3/Q4/Q5 curved CUDA actions
for both geometries, preallocation rejection, changed-thickness/criteria
rejection, CPU mapped spectra and the existing mounting/native pulse tests.
An earlier 17-test pass is retained. These overlap existing test groups and
are not added to a cumulative integrated-test total.

Native/GPU jobs peak at 524.47 MiB combined process-tree RSS, system RAM 52.7%,
minimum available RAM 7,671.13 MiB and observed GPU use 1,072 MiB / 53 C. Final
tests observe 466.40 MiB RSS and GPU use 1,074 MiB / 54 C. GPU readings include
other desktop use. CPU temperature is unavailable. Resource guards remain
active, and no training runs.

The independent audit recomputes forces from retained native matrices,
connectivity and states; reconstructs native-dual forces; checks every small
GPU matrix column; and recomputes modal solutions, errors and energy ranges
from saved final states. Its source and receipts are retained. The final
PNG/PDF was visually checked after suppressing overlapping minor x-axis
labels. CuPy emits its existing CUDA-path-detection warning, but all kernels
compile and execute; no global toolkit or PATH changes are made.

Run a bounded reference with:

```powershell
.\.venv\Scripts\python.exe -m examples.gpu_mapped_reference --order 3 --angular-cells 1 --output outputs/new-mapped-gpu-reference
```

Use `--intact` for its control; orders 3/4/5 and angular counts 1/4 are
explicitly bounded. Plans bind source/runtime, geometry and acceptance
criteria. Receipts seal element/global/GPU proof files and source snapshots.
Raw jobs are `outputs/mapped-gpu-{recess,intact}-q*-*-20261111`. Audit tools
and final figures are under `outputs/mapped-gpu-tools-20261111` and
`outputs/mapped-gpu-audit-20261111`. Compact evidence is
`evidence/mapped-recess-gpu-debug.json`, fingerprint
043709a7a0c6937dc76b676a64f28ee23fa81a9c06f12e4ba16b74efbe89474b.

The subsequent [surface-port checkpoint](mapped-surface-ports.md) connects
physical disk integration and actual per-state mounting geometry to this
backend, with complete small pulse tests. Full-size mapped mesh/capacity and
3.175 mm intact/recess space/time/frequency convergence,
measured removable-clamp coupling/contact and physical PZT/electronics
agreement remain required. This checkpoint accepts no research waveform
bank: every reference retains `ready_for_training_data=false`.
