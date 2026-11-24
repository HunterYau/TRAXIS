# MOR logic audit and corrections

The local implementation now distinguishes a model that fails the pole-growth
acceptance screen from a failed numerical calculation. These changes do not
establish a successful reduced model for the full Wandowski plate.

## Confirmed problems fixed

- The worker caught every `ArithmeticError` from propagation and reported it as
  an ordinary rejected candidate. Nonfinite calculations must instead remain
  visible failed stages. Only `ReducedModelGrowthError` is now handled as a
  scientific rejection.
- The growth screen previously prevented all waveform comparisons for a rejected
  candidate. The worker now evaluates the unchanged recurrence in an explicitly
  labelled diagnostic mode, saves and prints the raw errors, and always excludes
  that candidate from selection. Nonfinite propagation still fails immediately.
- Partial SVD validation scaled every residual by the largest singular value.
  This could hide an inaccurate weaker retained direction before ERA's inverse
  singular-value scaling. The original 1e-9 global test is retained and a 1e-7
  per-triplet relative test is added. Returned shapes and finite values are also
  checked before constructing the realization. This extra check is not evidence
  that the earlier H200 SVD caused the waveform failure.
- The unlaunched stable-separation trial omitted per-task artifact checksums
  required by its downloader. Its receipts now include them. That trial remains
  an experimental altered approximation, not a validated physical correction.

## Evidence from the saved H200 model

The 2048-state model has 674 poles outside the unit circle, maximum radius
1.0003337074074927. Its retained singular values range from 0.09241706529804856
to 0.0004915785004957152. The omitted Hankel Frobenius fraction is about 1.7073%.
This matrix approximation diagnostic is not a bound on voltage error.

The first original Markov vector is exactly `[0, 0]`, whereas the reduced model's
`C @ B` is approximately `[-9.3034e-7, -3.8278e-7]` V. This shows immediate
reconstruction error before the original signal starts. It is not an indexing
offset correction: the same h[2]=CB convention recovers independent small
state-space and production Verlet systems correctly. Finite-record truncation
can lose both delay cancellation and stable oscillatory dynamics.

No confirmed transpose, Hankel shift, convolution offset, or ERA formula bug was
found. Removing unstable modes would change the response and cannot be assumed
to fix these errors. The large plate ROM remains unvalidated.

## Verification and execution status

40 focused CPU tests passed across ERA, matrix-free ERA, broadband reconstruction,
and the existing spectral-separation helper. Regression tests include a weak
singular direction whose inaccurate triplet passed the former global residual
test, a growth-rejected model with exact diagnostic waveforms that must still
remain ineligible, and an unrelated arithmetic failure that must remain fatal.

No cloud action or full-size simulation was performed during this correction.
Changes are local; the deployed code and previously saved results are unchanged.
Existing frozen plans deliberately reject changed source hashes. Use a fresh
output root and `--prepare` before a separately requested full rerun, for example:

```powershell
.\.venv\portable-runtime\python.exe -m examples.pzt_mor_study --root outputs/simulation-runs/mor-logic-fixes-v5 --prepare
```

The physical full-order solver, mesh, timestep, precision, source, and 0.1% raw
waveform acceptance target were not changed. A successful corrected full-plate
ROM has not yet been demonstrated.

## Subsequent H100 NVL run

The changes above were subsequently uploaded and run on the H100 NVL. All nine
scheduled stages completed, and the downloader verified the results locally in
`outputs/simulation-runs/runpod-h100-nvl-mor-005`. The preceding local-only status
describes the audit checkpoint before this deployment.

All seven candidate orders failed the growth acceptance screen. Worst raw
waveform errors across the 50–250 kHz references and four voltage traces were:

| States | Worst relative L2 error (%) |
| ---: | ---: |
| 32 | 124.921 |
| 64 | 158.675 |
| 128 | 573.416 |
| 256 | 3881.72 |
| 512 | 35748.7 |
| 1024 | 254594 |
| 2048 | 136237 |

The target remains below 0.1%. These diagnostic errors compare MOR with the
full-order simulation, not with experimental measurements. No candidate was
accepted. The checks and reporting corrections did not resolve the underlying
full-plate reduced-model instability. The spectral-separation helper was not
used in this run. Generated waveforms and model archives remain in the ignored
local output directory; this table preserves the outcome in version control.
