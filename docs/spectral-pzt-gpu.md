# Optional spectral PZT GPU backend

The spectral experiment accepts `--backend gpu`. Its default remains CPU:
the measured six-ring component run was faster on the local CPU.
Q2 still defaults to six radial cells. The GPU option also supports Q3.

GPU operations: FP64 batched element quadrature and consistent mass integration,
exact sparse triangular solves, and equation residual evaluation. Global sparse
assembly and full pivoted SuperLU factorization remain on the CPU. There is no
global dense matrix conversion, approximate factorization, mass lumping, fitted
phase correction, or change to geometry, quadrature, material tensors, electrodes,
or input displacement. The original CPU path remains available.

CuPy documents CPU factorization with GPU triangular solves:
https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.splu.html
The implementation uses the same mechanism through `SuperLU` to avoid uploading
the matrix solely to download it again for CPU factorization.

## Local commands

From this worktree, using a Python installation with CuPy and CUDA libraries:

```powershell
python -m examples.spectral_pzt_experiment --backend gpu --radial 6 --output outputs/q2-gpu
```

For this computer's existing isolated CUDA dependencies, append:

```text
--local-cuda-root C:/Users/bobty/PycharmProjects/HelloWorld/TRAXIS
```

Its embedded Python requires `-c "import sys,runpy; sys.path.insert(0,'.'); runpy.run_module('examples.spectral_pzt_experiment',run_name='__main__')"`
instead of `-m examples.spectral_pzt_experiment`.

To launch the 6/8-ring schedule with the text console and saved outputs:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/run_q2_convergence.ps1 -Backend gpu
```

CPU and GPU filenames differ so results cannot silently reuse the other backend.
This remains an isolated PZT component experiment; the new backend is not an LTS
plate attachment. No full plate simulations were launched for this work.

## Verification and performance

Local RTX 3060 Ti: Q2 and Q3 small meshes were assembled with both backends;
relative matrix differences were at most 1.05e-15. All six frequencies and both
S0/A0 voltages on Q2 six rings matched saved CPU voltages with a worst change of
4.94e-12 percent. The same 1e-9 equation residual requirement applies.

First GPU component run: 29.67 s. A subsequent CPU/GPU sequence measured
6.75 s CPU versus 40.14 s GPU (assembly 1.60 versus 12.48 s). Startup, CUDA
compilation/cache and transfer overhead are included; these are not warmed
steady-state or whole-plate speed estimates. GPU memory pool after the parity
check was about 1.14 GiB; GPU host process peak in the repeat was 1511 MiB.
No GPU speedup is established, so CPU remains the default.

Receipts are in `outputs/spectral-pzt-gpu-check-001/verification.json`,
`timing-repeat/timing.json`, and `solver-sha256.json`. SHA-256 equality with the
original quadratic worktree plus an empty Git diff against HEAD verified six
original LTS and PZT files unchanged.
