# Independent quadratic PZT dimension refinements

The user-requested study reuses the saved **6/64/2** receiver result and
changes only one subdivision count in each new experiment:

| Dimension | Radial | Angular | Thickness | P2 nodes | Tetrahedra |
|---|---:|---:|---:|---:|---:|
| Saved reference | 6 | 64 | 2 | 7365 | 4224 |
| Radial | 8 | 64 | 2 | 9925 | 5760 |
| Angular | 6 | 80 | 2 | 9205 | 5280 |
| Thickness | 6 | 64 | 3 | 10311 | 6336 |

These remain isolated, frequency-domain receiver tests, not plate transients.
Every case prescribes the same analytical S0 and A0 plate-surface motions at
50,100,150,200,250,300 kHz, with the same material, electrical port and 1 nm
input scale. No gain, phase or delay fitting is used. Complex voltage
differences include both magnitude and phase. The saved reference is not
exact ground truth; one perturbation per dimension does not prove convergence.
Changing angular subdivisions also changes the polygon approximation to the
circular disk. This effect is included, not mistaken for a pure interpolation
order change.

## Resources and equation ordering

The initial 8192-node cap would reject all three configurations. An explicit
component-only override allows up to 12000 nodes and a 2304 MiB predicted
working set, still requiring a forecast below 55% of currently available
RAM. The full sparse assembly forecast remains bounded independently. The
revised forecast uses the saved reference's measured whole-process peak RSS,
scaled by nodes^2.5 with a 50% margin, and at least twice the explicit assembly
forecast. This is a planning estimate, not a mathematical memory bound.

Runs are serial on the local CPU. There is no GPU or RunPod work. Solver,
element integration, constitutive matrices, time-step settings and existing
simulation scheduling are unchanged. The override applies only to this
isolated experimental runner; the original lower default remains available.

The first radial attempt with COLAMD caused substantial sparse factor fill
and was interrupted after about 160 seconds, before a final results receipt.
Its attempt record is preserved. The completed study uses the symmetric
MMD_AT_PLUS_A equation ordering. A small actual PZT verification at 50,150,
300 kHz matched COLAMD solutions to <2e-15 relative difference with both
residuals <2e-15. This only permutes equations; it changes no physical model.
The original COLAMD ordering remains the default/fallback. Reference timings
and new timings consequently are not a controlled mesh-only speed benchmark.

## Commands

From this experimental worktree, in an activated Python environment:

```powershell
python -m examples.quadratic_pzt_basewaves --rings 8 --sectors 64 --layers 2 --output outputs/quadratic-pzt-dimensions --max-nodes 12000 --max-working-mib 2304 --memory-reference outputs/quadratic-pzt-audit/basewaves/r6-a64-t2.json --ordering MMD_AT_PLUS_A
python -m examples.quadratic_pzt_basewaves --rings 6 --sectors 80 --layers 2 --output outputs/quadratic-pzt-dimensions --max-nodes 12000 --max-working-mib 2304 --memory-reference outputs/quadratic-pzt-audit/basewaves/r6-a64-t2.json --ordering MMD_AT_PLUS_A
python -m examples.quadratic_pzt_basewaves --rings 6 --sectors 64 --layers 3 --output outputs/quadratic-pzt-dimensions --max-nodes 12000 --max-working-mib 2304 --memory-reference outputs/quadratic-pzt-audit/basewaves/r6-a64-t2.json --ordering MMD_AT_PLUS_A
python -m examples.quadratic_pzt_dimensions_report
```

Use a fresh output folder to repeat a case: existing results are never
overwritten. The reporter reads existing results and verifies that the saved
reference's SHA-256 matches the recorded schedule. Complete receipts,
per-frequency complex errors, runtime, sampled RSS, ordering parity checks,
and source hashes are retained under
`docs/evidence/quadratic-pzt-dimensions/` after completion.

## Completed findings

All three cases finished, with the saved reference reused unchanged:

| Refinement | S0 six-frequency L2 difference | A0 six-frequency L2 difference | Maximum A0 difference | CPU wall time | Sampled peak RSS |
|---|---:|---:|---:|---:|---:|
| 8/64/2 radial | 0.38429% | 1.13019% | 6.77133% | 70.80 s | 806 MiB |
| 6/80/2 angular | 0.01316% | 0.14055% | 0.88932% | 37.30 s | 527 MiB |
| 6/64/3 thickness | 0.22761% | 0.47399% | 2.90408% | 1247.77 s | 2875 MiB |

The largest A0 differences occur at 150 kHz, with small net voltage and
cancellation sensitivity. Radial refinement produces the largest response
change, thickness is second, and angular is smallest in this experiment.
This is evidence of directional sensitivity, not proof of true solution
error or full convergence. No new optimal mesh or acceptance is selected.

The thickness case took about 20m48s and exceeded its RAM forecast (1995 MiB
forecast vs 2875 MiB sampled peak), while remaining within the computer's
available memory. Future forecasts should account for this underestimate.
No precision, matrix integration, or physical parameters were reduced.
Resource estimates are admission forecasts, not runtime memory hard stops.
The earlier, aborted radial attempt is not included in the successful-case
wall time; its 160-second attempt record is preserved separately.

## GPU follow-up deferred

The CPU study finished before a GPU switch could be made. An isolated
cuSolver library was installed in the main workspace's ignored
`outputs/quadratic-pzt-gpu-libs` folder; the virtual environment was not
modified. A separate optional probe compared FP64 CPU LU and GPU sparse QR
on two small PZT systems. Solutions matched within 1.7e-14 relative error,
but the measured GPU runs were slower (these are not controlled, repeated
performance benchmarks). Raw probe receipts are saved separately.

The user then deferred GPU implementation. No production GPU backend was
enabled, and no additional study runs were scheduled. The standalone probe
script is retained for reproducibility; it is not used by the study runner.
CuPy's `splu` would not accelerate the factorization itself, as documented
in the [CuPy sparse LU reference](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.splu.html).
