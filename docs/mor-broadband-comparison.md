# Matrix-free MOR and broadband comparison

The default scheduler now uses **FFT Hankel products and PROPACK partial SVD**.
It does not allocate the dense 65,534 by 32,768 Hankel table. The earlier dense
ERA builder is retained in `examples/pzt_era.py`, with the 64-bit CUDA SVD
wrapper in `examples/pzt_era_cuda.py`, for controlled comparisons.

The new builder is `examples/pzt_era_matrix_free.py`; the scheduler remains
`examples/pzt_mor_study.py`. Original plate physics, full-order direct and
broadband methods, saved recordings, and earlier attempts are preserved.

## Established numerical methods

[Minster et al., section 2.2](https://arxiv.org/html/2003.11872v2#S2.SS2)
describe using FFTs for Hankel matrix products without storing the full matrix.
This implementation uses zero-padded linear convolution and vector reversal to
calculate those products for both H0 and shifted H1, including their transposes.
Both output channels are interleaved exactly as in the dense implementation.
Only batches of eight columns are processed at a time.

[SciPy PROPACK](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.svds-propack.html)
computes the largest required singular triplets through a LinearOperator.
This is Lanczos bidiagonalization, **not** the randomized SVD algorithm in the
Minster paper. A fixed starting-vector seed makes the calculation repeatable.
FP64 arithmetic and the complete original recording are retained. No downsampling,
filtering, source fitting, time shift, gain correction, or artificial damping is
introduced. Iterative SVD has numerical tolerance and must be checked; equality
with dense results is not assumed.

The ERA equations remain those of
[Juang and Pappa, NASA 1985](https://ntrs.nasa.gov/api/citations/19850022899/downloads/19850022899.pdf):

```text
m[0] = h[2] = CB; h[0] = h[1] = 0
H0[i*p+output,j] = m[i+j,output]
H1[i*p+output,j] = m[i+j+1,output]
H0 = U S V^T
Ar = S^(-1/2) U^T H1 V S^(-1/2)
Br = S^(1/2) V^T[:, :1]
Cr = U[:p, :] S^(1/2)
```

Only the leading requested triplets are computed. H1 times the retained vectors
and the ERA projection are evaluated in small column batches; H1 is never formed.
The exact full H0 Frobenius norm is computed from anti-diagonal multiplicities.
Tail diagnostics therefore include omitted singular values instead of incorrectly
reporting a zero tail from a partial spectrum. They are diagnostics based on
approximate singular values, not rigorous error bounds. Full numerical rank is
explicitly unknown.

## Convergence and memory

PROPACK tolerance is `1e-12`. Its basis dimension is explicitly limited to
`min(m+1,n+1,2*k+32)`, instead of its much larger default `10*k`. There is no
silent rank reduction or dense fallback on nonconvergence.

Both singular equations are checked after the solve. Their largest column
residual divided by the largest singular value must be below `1e-9`.
Both retained singular-vector Gram matrices must differ from identity by less
than `1e-7` in every entry. Failures stop visibly. These numerical checks do not
replace the final receiver-voltage comparison.

At 65,537 samples, two outputs, and maximum order 2048, the workspace estimate
is approximately **6.76 GiB host RAM and 0.30 GiB GPU memory**, versus the old
112 GiB GPU / 48 GiB host estimate. These are estimates rather than measured
full-run peaks or guarantees. The Lanczos bases live on the CPU, while the
optional GPU backend accelerates FFT products. GPU utilization need not stay
at 100%. Actual speedup has not been measured.

A NumPy FFT backend is implemented for CPU use and small correctness checks.
The prepared cloud schedule explicitly chooses CuPy FFTs. The full-size CPU
build has not been started locally; this computer currently needs more free RAM
than was available at the last hardware check to satisfy the conservative screen.

## Unstarted schedule

The new frozen root is
`outputs/simulation-runs/runpod-mor-comparison-matrix-free-v3`.

1. Full-order broadband 300 kHz, reconstructed from the saved impulse response.
2. One matrix-free ERA build at the saved **5.340576171875 ns** internal step.
3. Orders **32, 64, 128, 256, 512, 1024, 2048**, each reconstructing
   **50, 100, 150, 200, 250, 300 kHz** in order.

The 350 microsecond recording, all internal time nodes, geometry, source,
PZT conditions, order checks and **less than 0.1% raw voltage L2 error** target
are unchanged. Top, bottom, S0-related sum and A0-related difference must pass
separately. Raw waveform and envelope Pearson correlations are also saved.
These combinations are not asserted to be pure modes.

Order selection uses only independent saved 50-250 kHz direct solves. The
additional 300 kHz pulse is declared synthetic: plate 1 has no measured 300 kHz
receiver/generator pair. Its full-order reference reuses the saved broadband
kernel, so its reconstruction time is not a fresh full-order solver benchmark.
No amplitude, time or polarity alignment is fitted. Zero reference norms cannot
pass. Unstable candidates are reported without pole clipping.

The plan freezes code hashes, inputs and method settings. Results are written
and printed after each frequency; combined findings update after each stage.
Completed checksummed stages are skipped on resume; interrupted attempts are
retained. The status heartbeat reports phase and elapsed time during iterative
SVD without inventing a completion percentage.

## Environment and later launch

The new lock is `tools/runpod/requirements-mor-matrix-free-linux-lock.txt`.
It extends the unchanged Python 3.12 / CUDA 12.9 baseline with pinned official
cuFFT and nvJitLink wheels. The matrix-free builder does not require GPU cuSOLVER,
cuSPARSE, or cuBLAS. Their earlier dense-method lock and setup script remain.

On a separately approved pod:

```bash
bash tools/runpod/setup_mor_matrix_free_linux.sh
.venv-mor-matrix-free/bin/python -u -m examples.pzt_mor_study --run
```

Setup performs tiny GPU FFT/transpose and one-state ERA checks, not the full
study. The retained dense setup explicitly uses the optional
`verify_transfer.py --gpu --dense-svd` smoke check instead.
No pod was created, started or resumed, and no full-size MOR calculation or
plate simulation was run while implementing this replacement. Earlier deployment
controls target earlier frozen studies and must not be reused unchanged.

## What this MOR does and does not establish

It reduces the fixed actuator-to-receiver voltage dynamics. It is not a mass/
stiffness Galerkin projection and does not recover full displacement, energy or
PZT stress fields. Second-order structure, passivity and infinite-time stability
are not claimed. Changing a plate, mesh, material, timestep, PZT placement,
actuator or electrical condition requires a new full-order response and ROM.
The first response of each new plate is not accelerated by this method.

A saved full-order broadband kernel already allows very fast pulse reconstruction.
MOR's build and online costs must be measured separately; a benefit over that
existing reuse case is not guaranteed. Mesh/PZT convergence and real-world
validation remain separate tasks.

## Verification and history

**32 CPU tests passed:** nine new matrix-free checks, fourteen existing ERA/
scheduler checks, and nine existing broadband checks. New coverage includes
forward/transpose FFT products against explicitly formed tiny matrices,
unequal dimensions and block boundaries, agreement with dense ERA at the same
rank, six independent driven state-space responses, a tiny production
consistent-mass Verlet model with nearly cancelling receivers, solver failure,
invalid singular vectors, and memory estimation without full-size allocation.
These are implementation checks, not full-plate accuracy or speed results.

The first approved H200 attempt completed the full-order 300 kHz reconstruction
and failed at the legacy large-matrix SVD call. Its partial results were
downloaded to `outputs/simulation-runs/runpod-h200-nvl-mor-001`, then the pod
stopped automatically. The unrun 64-bit dense retry and its source snapshot are
also retained. The matrix-free plan has a fresh folder and source snapshot.
