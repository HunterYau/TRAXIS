# Q2 spectral PZT plate convergence

The existing PZT mesh study accepts `--pzt-element spectral` during preparation.
The original quadratic tetrahedral choice remains the default and is retained.
The opt-in Q2 schedule runs:

| Order | Annular radial cells | Polygon sides | Thickness cells | Nodes per PZT | Plate plus three PZTs DOFs | GPU memory forecast |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| New spectral reference | 6 | 64 | 1 | 7875 | 28217268 | 30.6 GiB |
| Spectral refinement | 8 | 64 | 1 | 9411 | 28226484 | 30.8 GiB |

The core grid has 16 cells per side. Geometry is the same 64-sided polygon as
the component studies. Both displacement and potential use Q2; Gauss order 6,
full consistent mass, material profile, ideal electrodes and perfect interface
tie are preserved. This replaces the PZT elements; the aluminum plate mesh and
existing LTS integration algorithm are unchanged. The study is radial-only;
it does not establish angular or thickness convergence.

The shared sparse attachment transfers forces and displacements with the same
transpose interface map, retains exact dielectric condensation as sparse factor
actions, and uses exact consistent mass factors. Body element eigenvalue bounds
are computed from Q2 matrices, not tetrahedral formulas. The existing requested
unchecked execution profile remains: coarse 10.68115234375 ns and fine
5.340576171875 ns; no half-step run. Thus temporal accuracy and stability of the
new full configuration are not certified by this preparation.

PZT assembly and factorization use the CPU; coupled matrix actions and the plate
evolution use the existing GPU sparse adapter. Full GPU factor solves are an
explicit option. CUDA graphs and the dense optimized adapter are not enabled.
Memory estimates are conservative planning values, not measured full-run peaks;
this full plate study is not sized for the local 8 GiB GPU. No reliable full-run
runtime estimate is established from the isolated frequency-domain component.

## Schedule and launch

Prepared locally under `outputs/simulation-runs/spectral-q2-convergence-001`.
The archived linear baseline is imported solely for fixed plate/source settings.
The first NEW spectral plate result becomes the comparison reference; isolated
PZT voltages and old tetrahedral plate data are not substituted as that reference.

General preparation (the baseline and archived assembly files must be available):

```text
python -m examples.pzt_mesh_study --prepare --pzt-element spectral --pzt-storage sparse --pzt-factor-backend gpu --execution-backend original --skip-stability-checks --root outputs/simulation-runs/spectral-q2-convergence-001 --baseline <archived-LTS-baseline> --assembly-sources <archived-assembly-sources>
```

After transferring the frozen project and prepared study to a suitable machine:

```text
python -m examples.pzt_mesh_study --preflight --root outputs/simulation-runs/spectral-q2-convergence-001
python -u -m examples.pzt_mesh_study --run --root outputs/simulation-runs/spectral-q2-convergence-001
```

No full plate or cloud execution was started for this change. The existing runner
retains kernel checkpoints and failure records, reconstructs six frequencies
50-300 kHz, and records per-channel raw/envelope/wiggle L2 differences, Pearson
correlations, arrival differences, runtimes, resource samples and plots. It
compares 8 rings against the new 6-ring reference at the 1% and 0.1% thresholds.
Results use a separate study folder. Existing RunPod download tools remain
available but are not started or configured by this local preparation.

## Small checks

`examples/spectral_pzt_coupled_check.py` independently constructs dense condensed
mechanical/electrical/mass operators on a tiny spectral mesh, then compares the
existing sparse GPU adapter on top/bottom faces with voltage/open ports.
Stiffness/local actions differed by at most 6.01e-16 relative; 16 original LTS
steps differed by at most 1.47e-17. FP64 and consistent mass were retained.
Receipt: `outputs/spectral-coupled-check-001/verification.json`.

Existing spectral and study tests: 19 run, 18 passed, one archived-fixture skip.
Protected solver hashes and Git diff verification are saved beside the receipt.
