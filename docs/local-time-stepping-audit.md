# Local time stepping theory and implementation audit

Historical audit of the former refinement study. Current settings and checks
are documented in [the correction](local-time-stepping-correction.md).

Reviewed against Grote & Mitkova, arXiv:1205.0654v2, section 3.1,
equations (3.13)-(3.17), LTS-LF2(p), and propositions 3.1-3.3:
https://arxiv.org/pdf/1205.0654 .

## Equation mapping

In mass-normalized coordinates z = M^(1/2) u, A = M^(-1/2) K M^(-1/2)
is symmetric positive semidefinite. All mass-connected body DOFs are included
in P, so P commutes with M and its square root. Therefore applying the same
recurrence in physical coordinates with M^(-1)K is equivalent, even though
that matrix need not be Euclidean-symmetric. Symmetry is in the M inner product.
This equivalence would fail if a fine projector cut a consistent-mass block;
the GPU adapter rejects that partition.

The implementation forms w = -M^(-1)K(I-P)u and q0 = 2u. Since the actuator
load and its mass inverse are wholly fine, the paper's coarse source term
(I-P)R vanishes. q1 uses the required half acceleration. Later inner steps
use the two source values at t +/- m*h and retain all rows of M^(-1)KP.
The outer result is q_p - u_previous. These factors, signs, source times,
input projection, and interface rows match the published recurrence.

At p=2 the effective operator is A_p = A - H^2 A P A / 16. An independent
closed-form test verifies that expression. At p=1 or P=0, the code reduces
to ordinary leapfrog. The CUDA ratio-one result agrees with existing Verlet
for supported zero initial conditions and force(0)=0.

## Stability and initial conditions

The adapter intentionally requires H <= 0.8 * the original full-system
stable step (where the original bound is 2/sqrt(lambda_upper)). It does not
use the relaxed coarse CFL range that requires additional LTS analysis.
For this conservative range the modified operator is positive semidefinite
and bounded by A: the inner Chebyshev filter has nonnegative spectrum with
2*(1-T_p(1-H^2*x/(2p^2)))/(H^2*x) between 0 and 1 for the admitted x.
Consequently H^2*lambda_max(A_p) remains below 4. Rigid translations/rotations
can give zero eigenvalues; positive definiteness is not assumed for a free plate.
This statement assumes the existing full-system stiffness/mass bound is valid.
It is not a proof of accuracy at a larger H, nor a claim that the fine-region
CFL restriction has been eliminated.

Only causal zero initial displacement/velocity and force(0)=0 are supported.
The smooth Hann excitation has zero startup value and derivatives. Negative
times use zero forcing. Nonzero initial states are not silently approximated.
The impulse kernels are numerical identification inputs, not continuous
physical impulses or evidence for the smooth-source convergence theorem.

## Broadband forcing

LTS is time invariant on coarse nodes, but periodically varying on the fine
input grid. Using the original single Verlet kernel would be wrong. There
are p kernels corresponding to fine indices 1 through p, each convolved
with its corresponding source subsequence. Independent tests at p=1,2,4
verify agreement with direct LTS integration to roundoff without any fitted
phase shift, time stretch, gain correction or cutoff.

The full schedule includes the unchanged original broadband control at H,
uniform fine Verlet at h, direct LTS, and LTS broadband at every frequency.
Timing accounts for ALL p kernels and all reconstructions. This matters
because multiple kernels can cancel the single-batch speed benefit.

## Audit corrections

The recurrence itself needed no equation change. The audit added forced
continuous-solution convergence checks, the explicit p=2 operator check,
actual CUDA LTS energy checks, saved full-run LTS energy diagnostics,
malformed-probe validation, interrupted-run handling, missing-result recovery,
scientific-plan checks, the original broadband control and explicit pipeline
timing. Older prepared plans are superseded by the fresh lts-003 plan.
An existing CPU-only Verlet test fixture also read uninitialized output even
when beta=0; 0*NaN could cause an intermittent failure. The fixture now uses
an overwrite in that case. The production Verlet solver was not changed.

Validation: all 27 focused CPU, CUDA, scheduler and original broadband
regression tests passed. Independent smooth forcing tests at substep ratios
1, 2, 4 and 8 reduced error by approximately four when H was halved, as
second-order theory predicts. A real CUDA p=2 impulse test conserved its
unforced discrete energy within the 1e-9 relative test threshold over 512
coarse steps. These were small verification systems, not full experiments.

## Limits

This audit supports mathematical and small-system implementation correctness
for the stated lossless model. It does not establish full-plate A0/S0 phase
accuracy, real-experiment accuracy, faster runtime, or Linux/Runpod deployment.
The full validation schedule has not been started. Damping, finite-resistance
ports, mapped/recess operators, moving/adaptive partitions and nonzero
initial conditions require separate implementations and checks.
