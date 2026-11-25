# Corrected acceleration setup

**Superseded schedule:** the current two-method, six-frequency benchmark is
documented in [local-time-stepping-benchmark.md](local-time-stepping-benchmark.md).
The three-run schedule below is historical.

This supersedes the previous lts-003 refinement study. That study kept
5.340576171875 ns globally and refined PZTs to 2.6702880859375 ns; it was
not an acceleration of the existing broadband calculation.

## Current settings

- Fine mechanical step: **5.340576171875 ns**, unchanged from the reference.
- Candidate coarse step: **10.68115234375 ns**.
- Duration: 350 microseconds. Reference: 65,536 steps; LTS: 32,768 coarse steps.
- Fine partition: complete consistent-mass PZT blocks, drive support, and
  explicitly requested regions. Exact interface coupling is retained.
- p=2 recurrence is algebraically reduced to one full stiffness action and
  one selected-element action per coarse step. No mesh, material or precision
  reduction, fitted phase adjustment or timing alignment is introduced.

## Published method and forcing choice

Grote & Mitkova, section 3.1, equations 3.13–3.17 and Proposition 3.3:
https://arxiv.org/pdf/1205.0654 . For p=2 the effective operator is
A_p = A - H^2 A P A/16, where A=M^-1 K in physical coordinates.

The acceleration scheduler uses **split-LFC forcing with nu=0**, as explicitly
specified in Remark 1 following Algorithm 1 of Grote, Michel & Sauter:
https://arxiv.org/html/2409.18085v1 . Set w=f(t_n)-A(I-P)u and gamma=0.
The drive is sampled at the COARSE time, while fine mechanical updates remain
at H/2. This is a published second-order source treatment, not equivalent to
sampling the drive at every fine substep. Its additional source/phase error
must be measured against the original fine-grid waveform. It permits one
time-invariant broadband kernel. The original fine-source option is retained
in the solver and still needs multiple phase kernels if used for broadband.
No stabilization parameter or claim of the stabilized method is added.

## Stability before propagation

The original unstabilized scheme can have unstable step-size bands; see:
https://doi.org/10.1090/mcom/3650 . Increasing H alone is insufficient.
The original full-system bound is enforced on h=H/2, ensuring positivity of
A_p for p=2. A preflight then screens the largest eigenvalue of H^2 A_p
in mass-normalized coordinates: the leapfrog limit is 4, and the implemented
rejection threshold is 3.8. Two independent ARPACK starts must converge with
residual <=1e-6 and agree within 1e-5. Maximum 256 GPU operator evaluations
total; lack of convergence rejects the run. Whole consistent-mass blocks use
a Cholesky change of coordinates. CPU ARPACK holds at most 12 basis vectors;
stiffness and inverse-mass actions run on GPU. Host RAM is screened first.

**This is a numerical spectral screen, not a rigorous eigenvalue upper-bound
certificate.** A converged Ritz residual does not prove that no larger
eigenvalue exists. Full-plate stability and phase accuracy remain unvalidated.
The simple conservative bare-plate safe bound is about 9.574 ns for this
mesh; 10.681 ns exceeds that sufficient bound. Therefore it must pass the
actual effective-operator screen; passing is not assumed from its name or
from a small test. Unforced energy drift is also checked during each run.

## Three full solves, not thirteen

1. Original broadband kernel at 5.340576171875 ns.
2. LTS broadband kernel with 10.68115234375/5.340576171875 ns steps.
3. Direct LTS excitation at the highest requested frequency, checking the
   single-kernel reconstruction independently.

Both kernels reconstruct 50, 100, 150, 200 and 250 kHz. Comparisons use exact
shared coarse nodes, with no interpolation or alignment; the full fine-grid
reference is also saved. Reports include raw voltage errors, signed wiggles,
S0/A0 sum/difference errors, envelope correlation and runtime. A predeclared
1% raw difference check is recorded for every channel; it is a numerical
agreement target, not a measured physical validation criterion. The direct
reconstruction check uses 1e-7 percent. Timing includes one kernel and all
reconstructions per method; stability-screen overhead is reported separately
and included in an additional end-to-end speed ratio. Faster execution is
not promised before measurement. Three tasks still mean three full solves.

## Verified and saved

33 focused CPU/CUDA checks passed locally, zero skips. These include independent
published-recurrence equivalence, smooth-source second-order convergence,
one-kernel reconstruction at multiple source frequencies, actual CUDA interface
actions, consistent-mass PZT coupling, comparison of the relaxed-step spectral
screen to the complete dense spectrum of a small real plate/PZT system,
energy conservation, scheduler/resume and original broadband regressions.
Logs: outputs/lts-corrected-checks.txt and outputs/lts-corrected-check-output.txt.
These are tiny verification problems, not a full-plate experiment.

The fresh lts-004 plan is preparation only. Old schema-1 plans are rejected
by the revised scheduler. No Runpod upload, pod action or full simulation was
performed for this correction. Next work must validate the actual full-plate
candidate; if unstable, inaccurate or slower it must not be called successful.

Commands from this worktree (execution remains a separate user decision):

```powershell
python -m examples.pzt_local_time_study --prepare --root outputs/simulation-runs/lts-004
python -m examples.pzt_local_time_study --run --root outputs/simulation-runs/lts-004
```
