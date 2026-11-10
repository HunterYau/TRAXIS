# Higher-order tensor-element investigation

The enriched tetrahedral full-plate thickness screen failed at 18.0%, 9.91%
and 19.56% raw finest waveform change. Refining only thickness may remain
expensive: an exploratory nominal 100 kHz / 3 mm periodic A0 check at 5 mm
lateral cells gives 2.95535% frequency error with four thickness cells and
2.97780% with eight. This motivates testing higher polynomial degree while
retaining the full spatial/temporal and physical acceptance requirements.

The tensor candidate is isolated on `research/tensor-element-reference`.
It does not replace the current forward backend or accept research data.
The earlier full-plate result remains reproducible at `fefe138` on
`research/enriched-spatial-refinement`.

## Formulation and quadrature

The spectral-element approach uses tensor-product Lagrange functions on
Gauss-Lobatto-Legendre (GLL) nodes. Positive GLL quadrature gives a diagonal
mass matrix. This is an established elastic-wave discretization; see
[Komatitsch and Tromp, 1999](https://academic.oup.com/gji/article/139/3/806/587264).
That publication does not establish suitability or performance on this plate.
The earlier enriched tetrahedral construction follows
[Geevers, Mulder and van der Vegt, 2018](https://arxiv.org/abs/1803.10065).

`traxis/tensor_hex.py` implements bounded affine Q1-Q4 hexahedral references.
Independent one-dimensional polynomial integrals produce tensor stiffness
blocks and consistent mass. Positive GLL mass is a separately named operator.
Two stiffness rules are explicit:

- `exact`: polynomial stiffness integration with the exact one-dimensional
  mass factors in undifferentiated directions.
- `gll`: standard GLL stiffness quadrature with its diagonal mass factors in
  undifferentiated directions. Derivative/cross factors have the quadrature's
  exact polynomial degree.

Both use the declared positive GLL mass for explicit-mode comparisons. The
consistent mass is retained for verification. The exact and GLL stiffness
matrices are not interchangeable: their Q4 Frobenius discrepancy is about
37.6% on the small fixtures. Independent native comparisons verify each
against its own declared rule. The difference is recorded, never hidden as
an exact-integration agreement claim.

`local_gll_action` separately applies gradient, constitutive stress and weak
divergence by tensor contractions without forming an element matrix. Tests
compare it with the full GLL stiffness on rectangular and skewed cells.
This supplies an independently checked operation for the subsequent
[accelerated backend](tensor-gpu-operators.md). Its CUDA implementation and
actual full-plate operator capacity are now measured separately.

## Geometry, continuity and mathematical checks

The small grid generator checks degree counts and a 128 MiB dense-workspace
screen before allocating coordinates. Dense reference assembly is limited to
2,000 vector coefficients. Tensor nodes are component-major with shared
interface nodes, positive affine determinants and explicit transform bounds.
No full plate is allocated through this small-reference API.

GLL weights and nodes pass polynomial moments through degree `2p-1` and
discrete integration-by-parts checks. The intentionally inexact degree `2p`
moment is also checked, preventing the diagonal mass from being described as
exact consistent mass. Other checks cover represented affine elastic energy,
six rigid modes with no extra zero modes, shared-interface mass, higher
coordinate moments, invalid orientation and rejection before allocation.

The first pure run failed because the installed NumPy root routine returned
complex dtype for real Legendre roots. The code now checks the imaginary
part against 1e-13 and explicitly extracts the real values. It does not
silently discard a materially nonreal root. The failed log/source are retained.
All six final focused tests pass without skips in 0.080 test seconds.

## Independent native operators

`examples.tensor_reference --mode operators` imports two connected hexahedra
into NGSolve and assembles all four operator types independently. Complete
Q1-Q4 native degree counts and component ordering are verified. The nodal
evaluation transform is checked at all GLL points, including shared-node
continuity and its inverse, before mapping native matrices to nodal form.

Eight cases cover four degrees on rectangular and affinely skewed 40-by-30-by-3
mm blocks. They pass separate exact-stiffness, consistent-mass, GLL-stiffness
and GLL-mass comparisons. Maximum relative block discrepancy is 4.26e-14.
All cases have positive GLL mass and exactly six rigid modes under the
declared eigenvalue screen. The largest native fixture has 675 vector
coefficients. These are operator references, not research-plate capacity.

The operator worker completes in 3.75834 monitored seconds at 336.51 MiB peak
combined RSS and 49.1% system RAM. The process-tree caps are 512 MiB and 180
seconds, with the usual 85% system-RAM and 2 GiB available-RAM limits. The
references are CPU-only; sampled GPU readings are system-wide observations.
CPU temperature remains unavailable on this PC.

## Periodic physical-spacing design study

`--mode dispersion` evaluates 432 declared combinations: Q2/Q3/Q4, 5/3.333/2.5
mm lateral cells, 1/2/4 thickness cells, 3/1 mm stock, 100/200 kHz, A0/S0 and
both stiffness rules. Exact Rayleigh-Lamb wavenumbers set the prescribed
phase; no nearby numerical frequency is selected after observing an error.

Opposite lateral faces are tied with the Bloch phase and interior coefficients
remain independent. Midplane parity separates A0 (odd x displacement, even z
displacement) from S0 (even x displacement, odd z displacement), excluding
SH motion. The independent sectors' cross-stiffness and Hermitian residuals
are checked. Maximum measured parity cross residual is 8.35e-16.

Lowest-band identification is used only when the target phase is inside the
first cell Brillouin zone. Seventy-two cases outside that range are explicitly
withheld; they receive no frequency error or passing label. They are not proof
that a higher-order element cannot resolve that wave. Broader-band mode
identification is still needed before choosing a mesh for the entire pulse
spectrum, including locally thinned regions. In particular, the carrier-only
results below do not settle the upper pulse bandwidth.

For the nominal 100 kHz, 3 mm plate at fixed 5 mm cells, standard GLL A0 errors
are:

| Degree | One thickness cell | Two thickness cells | Four thickness cells |
| --- | ---: | ---: | ---: |
| Q2 | 6.52291% | 4.68725% | 4.41469% |
| Q3 | 0.265193% | 0.147768% | 0.144976% |
| Q4 | 0.00299911% | 0.00271210% | 0.00270564% |

At 200 kHz in 1 mm stock with 2.5 mm cells, Q4 errors are 0.0146874%,
0.0145532% and 0.0145499%. This is a uniform-thickness mode comparison,
not a recess-scattering calculation. All sixteen finest combinations
(Q4, four thickness cells, 2.5 mm lateral cells) meet the predeclared 0.05%
frequency design screen; maximum relative error is 0.000145499, or 0.0145499%.
That screen does not replace the unchanged 5% full-window waveform gate.

The dispersion worker completes in 5.49580 seconds with 259.61 MiB sampled
combined RSS, 48.7% system RAM and at least 8,325.29 MiB available. It changes
no physics parameters, waveform window or acceptance criterion in the
earlier failed tetrahedral experiment.

## Evidence and next measurement

```powershell
.\.venv\Scripts\python.exe -m examples.tensor_reference --mode operators --output outputs/tensor-operators
.\.venv\Scripts\python.exe -m examples.tensor_reference --mode dispersion --output outputs/tensor-dispersion
```

Measured roots are `outputs/tensor-native-operators-20261109` and
`outputs/tensor-dispersion-design-20261109`. Plans bind sources, the native
runtime, material, criteria and source snapshots. Native comparison matrices,
evaluation transforms and finest generalized eigenproblems are retained.
The independent collector rechecks raw matrix discrepancies, positive mass,
rigid modes, parity/phase eligibility and every finest eigenproblem.

`evidence/tensor-reference-debug.json` binds the runs, raw audits, resources,
six-test group, retained first failure and the exploratory enriched comparison.
Its fingerprint is `3bdf90ec70361ed329b7450b7c702b03e1894df9fbb073bc577f4585a17a27aa`.
The collector and visually checked PNG/PDF design figures are in
`outputs/tensor-tools-20261109`. Only the nominal A0 within-zone subset of the
earlier exploratory study is used; its other untracked coarse branches are
not treated as accuracy evidence. The prior complete integrated suite remains
268 tests; focused counts are not added to it.

The prospective full 100-by-100-by-1 Q4 grid has 2,412,015 vector coefficients;
two/four thickness cells have 4,341,627 / 8,200,851. Those are exact algebraic
counts, not measured capacity or runtime. The subsequent
[profile-band study](lamb-profiles.md) assesses folded modes by analytic
displacement support and shows that the 5 mm carrier-only result fails the
sampled band screen. The subsequent GPU investigation verifies the accelerated
action and actual 150-by-150-by-1 Q4 full-plate operator capacity. The next
implementation needs measured complete-window cost, physical circular
load/probe integration and
full space/time comparisons. Conforming recess geometry, bonded/calibrated PZT
response and physical agreement remain required. No research bank or full
model/policy training is accepted by this work.
