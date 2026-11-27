# MOR research review after the H100 failure

## Findings

No new ERA formula, transpose, impulse offset, or convolution bug was confirmed.
The implementation is conventional finite-record, input-output ERA, not a
structure-preserving reduction of the plate mass/stiffness operators. Correct
implementation of this algorithm does not imply that its truncations are
accurate or stable for the current undamped plate.

The earlier explanation that pole growth alone caused every error was too
strong. Truncation already changes the initial response, and growth compounds
errors later. This review does not uniquely isolate the full-plate cause.

## Relevant research and remedies

1. Pelling and Sarradj, *Adaptive Reduced Order Modelling of Discrete-Time
   Systems with Input-Output Dead Time* (2025), Sections 2.4 and 3:
   <https://arxiv.org/html/2506.08870v1>.
   Propagation delays require additional states and can cause pre-arrival
   ringing when compressed. The authors separate input/output dead times,
   identify the remaining dynamics, then retain those delays explicitly.
   Their stability/error statement assumes the response has decayed; its
   assumptions cannot be transferred to our nondecaying record. Their acoustic
   experiments do not establish a cure for our dispersive piezoelectric plate.

2. Rezaian, Huang, and Duraisamy, *Non-intrusive Balancing Transformation of
   Highly Stiff Systems with Lightly-damped Impulse Response* (2021), Section 6:
   <https://arxiv.org/html/2109.10408v2>.
   Short observation horizons missed slow dynamics and led to inaccurate or
   unstable ERA models. Longer records resolved the decay. They also used
   output domain decomposition and sampling appropriate to each region.
   More samples per second and a longer physical observation horizon are
   different controls. Extending our lossless record will not make it decay.

3. Beddig et al., *Structure-Preserving Model Reduction for Dissipative
   Mechanical Systems*: <https://arxiv.org/abs/2010.06331>.
   Reviews mechanical reductions that retain second-order structure. This is a
   distinct implementation route requiring access to mechanical operators or
   suitable state data. Not every second-order reduction preserves stability;
   hypotheses and projection must be checked. It is not implemented by our ERA
   or by the experimental stable/unstable spectral-separation helper.

## Independent implementation checks

New regression file: `tests/test_pzt_era_research_audit.py`.

- Builds an eight-state undamped rotation system with SciPy `dlsim`, then adds
  an exactly known 16-sample delay. The original impulse response is bounded.
- Independently constructs H0/H1 with array slices, performs NumPy SVD, applies
  published ERA equations, and uses `dlsim` to propagate the resulting model.
  It does not use production Hankel assembly, ERA fit, or propagation for the
  reference calculation. The six-state production matrix-free response agrees
  with that reference, including its artificial growth.
- A 24-state dense realization recovers the delayed response; an eight-state
  matrix-free realization with the known delay kept separately also recovers
  it. These are test fixtures, not modifications to the plate scheduler.

| Independent fixture | States | Spectral radius | Raw relative L2 error (%) |
| --- | ---: | ---: | ---: |
| Compressed delayed response | 6 | 1.0011507765 | 33.8539749 |
| Full-rank delayed response | 24 | 1 + roundoff | 7.43e-12 |
| Known delay separated | 8 | 1 + roundoff | 4.22e-11 |

The toy test supports the mechanism and the code equations; it cannot assign
the full plate's error to delay alone. A stronger 24-state partial SVD on this
fixture was rejected by existing residual checks; those checks were retained.

43 focused CPU tests passed (ERA, matrix-free ERA, broadband, spectral split,
and the three new research checks). No new full-size solve or cloud action ran.
Production physics, timesteps, meshes, scheduler, and growth rejection remain
unchanged. No correction is claimed for the failed full-plate model.

## Saved plate evidence

The saved impulse's RMS in its final quarter is 82.5% (top) and 89.0% (bottom)
of the RMS in its second quarter. Equal duration windows were compared. This
is a voltage persistence diagnostic, not a mechanical energy measurement; the
tail is plainly not negligible at 350 microseconds.

At the actual integration timestep, top-channel onset varies from node 61
(any nonzero value, 0.326 microseconds) to node 3839 (1e-12 of peak, 20.50
microseconds) to node 7796 (1% of peak, 41.64 microseconds). Hence physical
arrival estimates cannot simply be declared an exact zero interval. Selecting
a delay threshold would discard small precursors and must be justified and
measured. Separate delays must preserve relative top/bottom timing and A0/S0
sums/differences, and must not flatten frequency-dependent dispersion.

The H100 build's maximum singular-triplet residual relative to its own singular
value was 6.73e-10; global residual 3.94e-12; orthogonality error 6.41e-12.
There is no positive evidence that an unconverged SVD caused the failure.

Machine-readable evidence is saved in
`outputs/result-analyses/mor-research-audit-20261123/evidence.json`.

## Next controlled comparison

Delay separation is a supported candidate, not a verified full-plate fix. A
trial must declare its delay rule, report discarded precursor norm and timing,
restore the exact chosen delays, and compare raw voltages and growth with the
unchanged broadband baseline. It must retain the original failed model as a
control. If the delayed lossless dynamics still fail, evaluate a mechanical
structure-preserving approach rather than assuming a longer record or pole
clipping guarantees accuracy. Larger plate/GPU trials await user approval.
