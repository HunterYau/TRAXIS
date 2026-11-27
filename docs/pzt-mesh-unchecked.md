# Two combined meshes, fixed time steps, screening disabled

Next-run preparation example: `outputs/simulation-runs/pzt-mesh-convergence-008`.
Completed run 007 retains its halved time grid and saved results.

| Order | Radial | Angular | Thickness |
| --- | ---: | ---: | ---: |
| Medium | 7 | 36 | 4 |
| Fine | 10 | 48 | 6 |

Each configuration uses one broadband kernel, reconstructed at 50, 100,
150, 200, 250 and 300 kHz. The saved baseline is reused. No half-step run.
The exact original time steps are restored: fine 5.340576171875 ns,
coarse 10.68115234375 ns (the requested rounded 5.34/10.68 ns pair).
Each kernel has 32768 coarse steps over 350 microseconds.
The previously completed run 005 retains its original time grid and results.

The study option `--skip-stability-checks` bypasses step-limit admission,
spectral screening, and energy-drift rejection. Energy drift is still
recorded. It uses the separate `traxis/gpu_local_time_unchecked.py` copy;
the original solver and its screening fallback remain unchanged. New plans
default to the optional optimized backend (memory reuse, CUDA graphs and
simple vector arithmetic fusion); use `--execution-backend original` when
preparing a fresh plan to select the old unchecked backend instead. Both
use exactly the time pair below and retain disabled stability screening.
The optimized backend receives `screen_stability=False` explicitly from
the study plan. It does not report a passed stability check. The
integration recurrence and FP64 arithmetic are unchanged. Output marks
stability as `performed=false, accepted=null`; it must not imply a pass.
Nonfinite states, invalid inputs, and memory guards still stop a failed
calculation. The scheduler stops after a failed configuration and retains
the pending configurations. No automatic pod stop is performed.

Preparation (from the study worktree, substitute the saved baseline paths):

```powershell
python -m examples.pzt_mesh_study --prepare --root outputs/simulation-runs/pzt-mesh-convergence-008 --baseline <saved-lts-005> --assembly-sources <saved-baseline-assembly-sources> --skip-stability-checks --execution-backend optimized
python -m examples.pzt_mesh_study --preflight --root outputs/simulation-runs/pzt-mesh-convergence-008
```

Execution, when requested separately:

```bash
python -u -m examples.pzt_mesh_study --run --root outputs/simulation-runs/pzt-mesh-convergence-008
```

Preparation and local checks do not launch the simulations or contact RunPod.
The export/download watcher also recognizes `stopped_after_failure`, so
failed-run records can be collected along with successful results.

The reported approximate element-count increase column is descriptive only; it is not an input to the mesh generator. Old configuration JSON files and prepared runs are retained as history and are not scheduled.
