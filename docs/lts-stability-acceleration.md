# Separate LTS stability-screening acceleration

Implemented in `traxis/lts_stability.py`. No edits to the existing integrator,
its effective-operator recurrence, the original stability checker, stiffness,
consistent PZT mass, source treatment, mesh, time steps or precision.
The scheduler alone now asks the separate module to pre-screen its system.

## Criteria retained

- FP64 throughout; ratio 2 and fine step <= 0.8 of the original full-system
  stable step for relaxed coarse stepping. Whole PZT mass blocks remain fine.
- Two starts, seeds 2718 and 3141, 32-vector basis, `tol=1e-8`,
  `maxiter=1000`, independently evaluated normalized residual <= 1e-6.
- The two estimates must agree within 1e-5 and their maximum plus maximum
  residual must remain strictly below 3.8 (the theoretical limit remains 4).
- CuPy can return its last approximation when its iteration allowance ends.
  The GPU path additionally checks its requested absolute 1e-8 tolerance;
  a nonconverged GPU result never silently becomes an accepted CPU fallback.
- A two-start Ritz result remains a numerical screen, not an upper-bound proof.

## Implemented paths

1. **GPU eigenvalue iteration.** CuPy's matrix-free thick-restart Lanczos holds
   the basis, orthogonalization, full vectors and exact Cholesky mass similarity
   on the GPU. Small PZT Cholesky factors are formed in FP64 once on the CPU;
   their triangular inverses and factors are applied on-device. Full GPU
   vectors no longer cross PCIe on each operator call. The projected small
   eigenproblem may still run on the CPU inside CuPy; this is not a claim
   that every scalar operation is on the GPU.
2. **Exact model caching across runs.** Keys hash actual mass, partition,
   geometry, device stiffness/mass/inverse blocks, tensor coefficients and
   kernel code, all `traxis` Python sources, numerical library versions,
   criteria, coarse step and substep ratio. Cached reports carry checksums and
   must satisfy the criteria again. Corrupt or mismatched reports are ignored.
   Hashing is streamed in 8 MiB chunks. No configuration label or old summary
   alone is treated as proof of matching operators.
3. **Avoid the repeated in-process check.** A successful optimized screen
   populates the original solver's existing per-system cache. Its original
   pre-integration call then reuses that result. No function is monkey-patched.
   Frequency, waveform length, probes and input amplitude do not change a
   fixed linear mechanical stability operator and need no separate screen.
4. **Warm starts.** If the operator and partition match but H changes, only
   the first start uses the previous normalized eigenvector, perturbed by
   0.001 times a normalized independent random vector. The second start stays
   random; both full solves and acceptance checks still run. Warm hints never
   count as certificates. Only one latest warm vector is kept, at most 256 MiB
   by default; set `max_warm_bytes=0` to disable saving hints. An exact key hit
   uses its certificate instead of doing a warm solve.
5. **Conservative bounds.** Cheap proofs can replace iteration when sufficient;
   see the derivation below. If insufficient they neither accept nor reject
   the candidate and the numerical screen proceeds normally.

The unchanged checker is available as `backend='original'` or the scheduler's
`--stability-backend original`. `auto` attempts GPU iteration only when both
its optional libraries and the forecast memory are available; otherwise it
uses CPU ARPACK with the same two-start checks. `gpu` explicitly requests GPU
and fails if unavailable. There is no automatic time-step or accuracy change.

## Conservative bound derivation

In mass-normalized coordinates put S=L^T (M^-1 K) L^-T, M=L L^T.
S is symmetric positive semidefinite, and the whole-block fine projector P
commutes with this mass transformation. For p=2 the unchanged method has

    B = H^2 S - (H^4/16) S P S = T - T P T/16,  T=H^2 S.

The original full-system certificate gives lambda_max(S) <= 4/dt_stable^2.
The unchanged fine-step restriction gives H^2 lambda_max(S) <= 10.24 < 16,
which establishes positivity of B under the same mass/stiffness assumptions.
Since T P T is positive semidefinite, B <= T. Consequently

    lambda_max(B) <= x = 4 (H/dt_stable)^2.

If this upper bound plus a floating-point safety margin is strictly below
3.8, the costly eigensolve is unnecessary. For P=I, B=T-T^2/16, so a sharper
upper bound is max(t-t^2/16) over 0<=t<=x: x-x^2/16 for x<=8, otherwise 4.
The original H<=0.8*dt_stable shortcut is retained unchanged. These are analytic
bounds under the original assembled PSD stiffness/SPD mass assumptions, not
interval-arithmetic certificates for every floating-point operation.

The recorded full plate has H/dt_stable about 1.162 and only 546,090 of
28,175,229 DOFs fine. The generic x is about 5.40, so this bound does NOT
eliminate its first screen. We do not assume it does merely because the
previous Ritz value was about 1.906.

## Resources and deployment

The GPU forecast reserves `(ncv+16)*n*8` bytes plus consistent-mass block
workspace, respecting both free VRAM and the system's device-memory budget.
At 28,175,229 DOFs this is about 10.8 GB plus patch workspace and existing
operator storage (previously screened around 4.6 GB). It is a cloud-sized
eigensolve and is not claimed to fit the local 8 GB RTX 3060 Ti. CPU fallback
also checks currently available host memory; it cannot automatically make a
full plate fit the local 16 GB RAM. Only tiny operators were exercised locally.

Native CuPy Lanczos needs cuBLAS and cuSPARSE in addition to the original
custom-kernel runtime. The original local environment lacked those libraries; the native verification
follow-up below installed isolated Windows libraries.
An optional, separate Linux x86_64 lock file is provided:
`tools/runpod/requirements-lts-eigensolver-linux-lock.txt` (cuBLAS 12.9.1.4,
cuSPARSE 12.5.10.65, nvJitLink 12.9.86). Their official PyPI wheel SHA256
hashes are pinned. The optional Linux downloads total about 987 MB and were not installed locally.
Windows libraries were subsequently installed in a separate target folder as
described below; the existing simulation environment remains unchanged.

After choosing to enable GPU screening in an isolated compatible CUDA 12.9
environment, install with `python -m pip install --require-hashes --no-deps
-r tools/runpod/requirements-lts-eigensolver-linux-lock.txt`. This is optional;
the original CPU checker remains usable. No pod was created, resumed, or
used, and no new full-plate simulation was executed.

Future prepared schedules use `outputs/stability-screen-cache` in this
worktree. Earlier saved reports lack the new operator fingerprints, so they
are not silently imported as exact cache certificates. The existing completed
run and its frozen plan are preserved; a future run needs a freshly prepared
plan because the scheduler/source snapshot changed. No new transfer archive
was built or uploaded.

## Validation

Targeted checks in `tests/test_lts_stability.py` and
`tests/test_local_time_schedule.py` cover:

- Conservative bounds against independently assembled dense spectra.
- The original CPU checker, separate CPU checker and GPU mass-normalized
  operator against a dense, genuinely coupled consistent-mass PZT plate.
- Exact disk-cache reuse and reuse by the unchanged original solver cache.
- Mass, stiffness, partition and time-step invalidation; corrupt reports/hints.
- Optional-library and low-VRAM fallback, original checker delegation,
  split-mass-block rejection, threshold/disagreement/residual rejection.
- GPU max-iteration returns that miss the requested tolerance are rejected.
- Real GPU operator actions driven by SciPy as a small reference eigensolver.

The last item validates the GPU adapter without claiming native CuPy Lanczos
was run at the initial checkpoint. The later native follow-up now tests CuPy
Lanczos directly, as recorded below. There were no full plate solves or
changes to the integration method.

## Estimated performance, not measured full-plate speedups

Reference: the downloaded completed run recorded 260 operator evaluations and
369.826 seconds in its original stability screen. Operator size was 28,175,229
DOFs, so one full FP64 vector is 225.4 MB. Avoiding one upload and download
per evaluation removes approximately 117 GB of repeated PCIe transfer at that
same evaluation count, in addition to moving bulk orthogonalization to GPU.

| Situation | Planning estimate | Qualification |
|---|---|---|
| Exact repeated model | Around 1-5 seconds on fast local storage | Fingerprinting/IO remain; zero eigen-operator evaluations. Network disks can take longer. |
| First GPU screen | Tentative 2-5x faster, about 74-185 seconds instead of 370 | Unbenchmarked; GPU memory bandwidth, convergence and CPU normalization/setup overhead determine the outcome. It may improve less. |
| Nearby-H warm start | Tentative 10-40% fewer total evaluations | Only the first start benefits; second independent solve retained. No guaranteed reduction. |
| Sufficient conservative proof | Zero eigen-operator evaluations | Does not currently prove the recorded relaxed full-plate case below 3.8. |

The original plate solve was about 415 seconds and LTS about 238 seconds.
A 1-5 second reusable check would leave LTS near 239-243 seconds excluding
shared construction, about 1.71-1.74x faster overall in that comparison.
These estimates do not claim faster wave propagation: the integrator and
its step count are unchanged. No measured GPU-screen speedup is reported.

## Primary references

- [Grote and Mitkova, LF2 effective operator and conserved energy, Propositions
  3.2-3.3](https://arxiv.org/pdf/1205.0654).
- [CuPy eigsh documentation: matrix-free operators, starting vector and
  thick-restart Lanczos](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.eigsh.html).
- [SciPy eigsh documentation: ARPACK, v0 and convergence settings](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html).
- [Official PyPI cuBLAS metadata](https://pypi.org/project/nvidia-cublas-cu12/12.9.1.4/),
  [cuSPARSE metadata](https://pypi.org/project/nvidia-cusparse-cu12/12.5.10.65/),
  [nvJitLink metadata](https://pypi.org/project/nvidia-nvjitlink-cu12/12.9.86/).

Solver hashes and before/after Git patch comparisons are recorded separately
under `outputs/stability-screen-audit/`; pre-existing uncommitted solver work
is preserved. Hash equality is against the start of this task, not a claim
that those files match Git HEAD.

## Native GPU verification follow-up

With the user's approval, installed the Windows x86_64 wheels for cuBLAS
12.9.2.10, cuSPARSE 12.5.10.65 and nvJitLink 12.9.86 into the main checkout's
`outputs/stability-screen-audit/cuda-eigensolver-libs`. The separate Windows
lock file pins official PyPI wheel SHA256 hashes. Installation used
`--no-deps --no-cache-dir --require-hashes --target`, preserving the existing
simulation .venv and system CUDA configuration. Download size was about
951 MB. Windows cuBLAS uses the most recent available Windows wheel; the
Linux optional lock remains independently pinned.

An isolated --target directory is not automatically found by CUDA Pathfinder.
`tools/runpod/lts_bundle/check_native_gpu.py` opens the four required DLLs
(nvJitLink, cuBLASLt, cuBLAS and cuSPARSE) by their explicit paths for the test
process only. It does not change Windows PATH or permanent environment settings.
Run it with the working project Python and `--cuda-libs` pointing to that
folder and `--output` pointing to a report directory. Its integrity checks use
the local baseline in this worktree's `outputs/stability-screen-audit`.

19 targeted checks passed with zero skips, including four native CuPy
Lanczos tests in `tests/test_lts_stability_native_gpu.py`. Native screening
on a 51-DOF genuinely coupled consistent-mass PZT reference produced:

- Dense lambda_max: 2.836733513077994.
- Original CPU estimates: 2.8367335130779976 and 2.8367335130779963.
- Native GPU estimates: 2.8367335130779967 and 2.8367335130779963.
- Maximum GPU/dense absolute difference: 2.6645352591003757e-15.
- GPU residuals: 4.42521150594269e-16 and 1.3762976753361245e-15.

The native checks also verified exact cached reuse with zero operator
applications; GPU warm starting at a changed step; recovery from a misleading
subdominant eigenvector hint; and rejection of an independently controlled
operator whose largest eigenvalue is 3.9, above the unchanged 3.8 threshold.
The previous mocked GPU-adapter check is retained but is no longer the only
GPU test. The new native cases run the actual CuPy eigsh implementation.

Platform: RTX 3060 Ti, CuPy 14.2.0, CUDA runtime 12.9, driver API 13.3.
All five protected solver SHA256 hashes and the protected before/after Git
patches still match. The detailed receipt is
`outputs/stability-screen-audit/native-gpu/report.json`; its `checks.txt`
contains the test log. The original pre-native receipt was preserved.

These small checks verify correctness paths, not full-plate performance.
The earlier 2-5x GPU speedup range remains an estimate; no full-size screen,
full simulation or Runpod activity was performed during this follow-up.
