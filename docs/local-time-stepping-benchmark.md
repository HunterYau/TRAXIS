# Current local time stepping benchmark

This replaces the earlier refinement and three-run schedules.

## Requested comparison

One intact plate, fixed geometry, material, mesh and PZT models. Two sequential
methods: normal broadband, followed by broadband with local time stepping.
Each produces 50, 100, 150, 200, 250 and 300 kHz voltage waveforms from a single
impulse solve. Exactly two full solves; no convergence sweep or separate
per-frequency solves. Normal broadband is the numerical reference. There is
no comparison with Wandowski measurements and no experimental data loading.
The existing plate/PZT configuration supplies physical parameters only.

Both methods use the identical prescribed five-cycle Hann sine excitation at
each frequency, with unit sine coefficient. This differs from the older
experimental drive's sample-peak normalization. The established normal
broadband solver and reconstruction formula are unchanged.

Fine mechanical step: 5.340576171875 ns. Coarse step: 10.68115234375 ns.
The fine partition contains complete consistent-mass PZT blocks, actuator
support and an automatic 4 mm edge strip (one plate element for this mesh).
Other regions use the coarse update. Optional fine boxes select additional
regions; they do not create defects or change plate geometry. The scheduled
plate has no defects. The adapter supports the affine tensor plate/PZT
operator; mapped/recess operators are rejected rather than treated as an
intact plate. No claim of supporting every defect geometry is made.

## Equations and checks

The mechanical recurrence is the published lossless LTS-LF2 at p=2:
https://arxiv.org/pdf/1205.0654 . The effective operator is
A_p=A-H^2 A P A/16. Its optimized implementation uses one full and one local
stiffness action per coarse update. Interface rows and exact PZT consistent
mass are retained. The drive treatment is the published split-LFC variant,
nu=0, described in Remark 1 after Algorithm 1:
https://arxiv.org/html/2409.18085v1 . Mechanical substeps remain H/2, while
the forcing is sampled at coarse times. This second-order source treatment
allows one broadband kernel; its error is included in the comparison.

The unchanged h must satisfy the original full-system stability bound.
Before either full solve, the actual full-plate effective operator must pass
the existing two-start numerical spectral screen, including consistent-mass
normalization, residual checks and a margin below the leapfrog limit. This is
a numerical screen, not a formal eigenvalue upper-bound certificate. The
paper warns of unstable step-size bands in the unstabilized method:
https://arxiv.org/abs/2005.13350 . Failed or inconclusive screening prevents
propagation; it does not silently change the requested steps.

Small tests verify the recurrence independently, second-order smooth-source
convergence, single-kernel reconstruction, selected-element/interface actions,
PZT consistent mass, edge and box selection, a relaxed-step screen against a
complete small-system spectrum, and the two-method six-frequency scheduler
and resume path. Full-plate stability, accuracy and speed remain unmeasured
until the benchmark is explicitly run. No full simulation or Runpod action
was performed while preparing this schedule.

## Outputs and execution

Each run has normal-broadband and lts-broadband directories, each containing
50khz through 300khz waveforms.npz files and source metadata. The normal
method completes all six reconstructions before the LTS solve begins.
Comparisons use exact shared 10.68115234375 ns nodes without interpolation,
time shifts, phase fits or gain matching. Full normal-method fine-grid
waveforms are also retained. Reports include raw relative voltage L2 error,
signed waveform Pearson correlation, normalized wiggle/envelope metrics and
top/bottom sum/difference traces. Correlation is not an accuracy percentage.
The raw error includes amplitude differences.

summary.txt prints readable error/correlation results; comparisons.json,
performance.json and all-results.txt retain full metrics and timings. Pipeline
times account for each kernel and its six reconstructions/output writes;
the extra LTS stability-screen cost is included in a separate speed ratio.
Completion means execution finished, not automatic validation of accuracy.

From the local-time-stepping worktree, preparation is separate from execution:

```powershell
python -m examples.pzt_local_time_study --prepare --root outputs/simulation-runs/lts-005
python -m examples.pzt_local_time_study --run --root outputs/simulation-runs/lts-005
```

The full mesh exceeds the local 8 GB GPU workspace screen. Preparation and
small verification tests fit locally; execution checks actual available host
and GPU memory and preserves the mesh and float64 precision on insufficient
hardware. No cloud deployment package is prepared by these commands.

## Stability-screen restart fix

The first H100 NVL attempt stopped before propagation because a separate
256-operator-call limit interrupted ARPACK. That did not establish instability.
The limit was removed, ARPACK workspace increased to 32 vectors (memory screened
before allocation), and maxiter set to 1000. The solver still requires tol=1e-8,
residual <=1e-6, two starts agreeing within 1e-5 and the same 3.8 acceptance
threshold below the theoretical limit 4. Progress is printed every 32 operator
evaluations. No time step, mesh, material, precision or physical recurrence was
changed. Both local and pod runs passed the 31 numerical readiness checks.
The failed attempt was preserved in outputs/attempt-001-preflight-cutoff on
the pod. The original transfer archive predates this patch; deployment.json
records the applied patch and the live project has an updated manifest.
