# Separate matrix-free Q2 PZT architecture

The selected design is partial assembly with tensor sum factorization for the
mechanical and electrical coupling actions. Retain the exact sparse dielectric
and scalar consistent-mass matrices and full sparse LU factors. Analyze each GPU
triangular solve once, keeping descriptors, fixed right-hand-side buffers and
workspace alive. Apply the same scalar mass/factor to three Cartesian RHS columns.
This is a cost-informed architecture choice, not proof of globally optimal speed.

Why this combination:

- Sum factorization avoids assembled body stiffness and coupling matrix reads.
  Geometry metrics and Jacobian-weighted Gauss weights are computed once.
- Exact dielectric elimination remains implicit; no large dense Schur matrix.
- Consistent mass inversion still uses the retained direct sparse factors. A
  diagonal GLL mass or unconverged iterative solve would change the formulation.
- One scalar coupled mass replaces three redundant Cartesian copies. Its LU
  solves all three axes together, with GPU analysis reused.
- The plate retains its existing tensor GPU kernel. The coupling uses exactly
  the existing surface interpolation and its transpose for force transfer.
- The new adapter plugs into original LocalActions; no LTS implementation is
  overwritten or edited. Buffer ownership and serialization follow its lock.

The existing sparse and dense/precomputed implementations remain references.
The GPU implementation explicitly supports the requested Q2 / six-point Gauss
rule, FP64, aligned perfect ties and disjoint mass supports. Unsupported cases
raise an error instead of falling back to a different model or CPU stepping.

Sources: [libCEED partial assembly and tensor evaluation](https://libceed.org/en/latest/intro/),
[NVIDIA cuSPARSE SpSM API](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespsm).
NVIDIA requires the analysis descriptors and workspace to stay consistent and
alive between analysis and solves; the new fixed-buffer plans retain ownership.

Only small local component/equivalence checks are authorized. No full plate
study, RunPod connection, upload, process change or automatic scheduler switch
is part of this implementation.

The installed CuPy 14.2 CUDA binding rejects cuSPARSE calls during stream
capture. Cached LU plans therefore use ordinary GPU launches. This restriction
is explicit in the adapter metadata. Tensor-only graph replay is supported and
checked; coupled CUDA graph replay is not claimed. No private binding or driver
patch is used to bypass that restriction.

## Implementation and usage

- `traxis/spectral_partial.py`: unchanged Q2 mesh, cached 1D basis/metrics,
  tensor reference actions, and scalar-only dielectric/consistent-mass assembly.
- `traxis/gpu_spectral_partial.py`: element CUDA kernels for K, B, B-transpose,
  dielectric and scalar mass actions. The hot correction path computes B*phi
  only, avoiding an unused dielectric application.
- `traxis/gpu_cached_lu.py`: persistent SpSM descriptors and buffers, two analysis
  calls per LU factor. Reuses exact SciPy-generated factors on the GPU.
- `traxis/gpu_spectral_matrixfree.py`: the original interface maps, transposed
  force transfer and ideal-open rank-one term, plus the existing plate operator.
  Mass uses one scalar sparse matrix/factor per PZT and three RHS at once.

Construct data with `spectral_partial.component`, then use the original
`SparseAttachment` and `ConsistentMass` constructors and call
`couple_matrixfree(base, attachments, masses)`. The returned system supports
the existing `LocalActions` and the study-only unchecked original LTS runner.
Close the system when finished to release native analysis descriptors safely.
It is serialized by the existing operator lock; shared work buffers are not
independent concurrent streams.

The new component requires explicit `rings`, `sectors` and `layers` arguments;
there is no implicit mesh downgrade. Pass the existing study row unchanged.

No existing study scheduler or frozen plan was modified to select this backend.
The original dense stability normalization/graph executor is not a compatible
adapter for these implicit operators. Full-system stability/temporal validation
and a new explicitly prepared study remain separate from these component checks.
The existing sparse adapter has the same disjoint-support restriction; overlap
raises an error rather than approximating the mass.

## Small local verification

Run CPU checks:

```text
python -m unittest tests.test_spectral_partial -v
```

Run the small GPU check (after loading the existing CUDA 12 dependencies):

```text
python -m examples.spectral_matrixfree_check --output outputs/spectral-matrixfree-check-001
```

Windows can add `--local-cuda-root <original-project-root>` to use its existing
isolated DLL directories. No libraries are installed by this script.

Verified on the local RTX 3060 Ti:

| Check | Largest relative difference |
| --- | ---: |
| Tensor stiffness, coupling, dielectric and consistent mass vs assembled reference | < 5e-16 |
| Manufactured quadratic bending force | 4.73e-14 |
| Cached LU with changing one/three RHS and nontrivial permutations | < 2e-16 |
| Both faces, driven/open ports and transformed mass/stiffness | < 5e-16 |
| Interface virtual work | 1.41e-15 |
| 65 original LTS steps, four derived voltage channels, raw/envelope | < 5e-16 |

Tensor-only CUDA graph replay also passed. Across repeated coupled calls the
three-PZT model retained two electrical analysis calls and six mass analysis
calls total (one lower/upper pair per factor), rather than analyzing every solve.
The tiny coupled model retained 1,158,324 PZT device bytes (about 1.10 MiB).
Five warmed full/local action pairs measured 23.41 ms for the original sparse
reference and 5.68 ms for the new path, about 4.12x on this SMALL local test only.
This is not a full-plate runtime prediction. Receipts are stored in
`outputs/spectral-matrixfree-check-001/gpu-check.json`.

For the actual 6/8-ring meshes, the per-Gauss inverse Jacobian and weight arrays
alone occupy about 10.55/12.66 MiB per SHARED component. Sparse mass/electrical
factors, analysis workspaces and patch buffers add to that; their full-size
GPU peaks have not been measured. No global dense PZT K/M/inverse or electrical
Schur matrix is allocated. The 28-million-DOF plate and LTS vectors still need
their existing memory, so the component figures do not imply that the full
simulation fits the local 8 GiB GPU.

This backend preserves the full original element eigenvalue-bound calculation
during initialization. It does not introduce approximate inverse solves,
mass lumping, reduced quadrature, mesh reductions, phase fitting or lower
precision. CPU assembly is limited to geometry and scalar M/D/factors; stepping
uses GPU arrays and cached direct solves.
