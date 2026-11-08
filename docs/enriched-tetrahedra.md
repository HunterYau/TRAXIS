# Enriched tetrahedral candidate for explicit elasticity

The current consistent-mass H1 reference is useful for accuracy checks, but
its measured full-size solve still needs many CG operator applications per
step. This separate candidate investigates a positive diagonal mass so that
explicit elastic propagation can avoid those solves. GPU propagation and
research-quality waveform acceptance have not yet been implemented here.

The 15-node construction follows Table 1 of
[Geevers, Mulder and van der Vegt (2018)](https://arxiv.org/abs/1803.10065).
It contains the complete quadratic space plus four cubic face bubbles and
one quartic cell bubble. The nodes are four vertices, six edge midpoints,
four face centroids and one cell centroid. The nodal integration weights are
positive. This is a deliberately different space and mass rule from ordinary
ten-node P2; the existing consistent-mass operators remain available unchanged.

## Independent construction and native availability

`traxis/enriched_tet.py` builds nodal basis polynomials, exact barycentric
gradient products, exact comparison consistent mass, and the positive nodal
mass. Geometry stays affine tetrahedral and the stiffness retains all three
elastic displacement components. Coefficients are component-major.

The installed NGSolve 6.2.2606 `H1LumpingFESpace` docstring describes only 2D
triangles. An isolated actual 3D probe instead returned 15 scalar functions
per tetrahedron and 51 global scalar coefficients on the six-tet control.
The docstring is stale for this tested build. No installation or native binary
change was needed. The probe and receipts are retained under
`outputs/mass-lumped-screen-20261107`.

The independent basis is checked against both the installed native enriched
space and a native P4 embedding. Those comparisons use explicit higher
quadrature to cover the enriched cubic/quartic functions. The comparison
consistent mass and the new positive diagonal mass are reported separately;
their difference is an intentional discretization choice requiring subsequent
dispersion and mesh/time validation, rather than operator equivalence.

## Completed numerical checks

Five pure checks cover the nodal identity, partition of unity, quadratic
reproduction, required quadrature moments, positive weights, derivatives,
independent Duffy/strain-matrix integrals, all 24 vertex permutations,
shared-face continuity and allocation/input limits.

The isolated native check covers rectangular and affine-skewed six-tet
controls with 153 vector coefficients each. Both native embeddings have full
column rank. Every stiffness and comparison-mass coefficient is compared,
along with the native nodal quadrature mass. The largest matrix discrepancy
is 4.10e-14. Both cases have six rigid modes and a conservative explicit
stability bound below the exact discrete eigenvalue limit. These are discrete
operator checks, not validation of full-size Lamb-wave propagation.

Manufactured unforced eigenmode traces are compared with their exact discrete
cosine solution. The initial 32/64-step rectangular error ratio was 4.239,
outside the fixed 3.8-4.2 asymptotic-rate screen. That failed coarse result is
retained. The same physical interval was then resolved at 64, 128 and 256
steps; ratios are 4.053/4.012 for the rectangular case and 4.028/4.002 for the
skewed case. The criterion was unchanged. Modified-energy relative ranges
remain below 1.36e-15. This manufactured short interval is a time-integrator
check, not a plate acquisition window or source/receiver experiment.

```powershell
.\.venv\Scripts\python.exe -m traxis.enriched_verification --output outputs/enriched-reference --max-seconds 90 --max-rss-mib 1024
```

The retained final-source comparison is under
`outputs/enriched-native-final-20261108`. Earlier coarse-failed and refined
attempts remain under `outputs/enriched-native-reference-20261107` and
`outputs/enriched-native-refined-20261107`. Plans, source snapshots, arrays,
case receipts and resource monitors are preserved.

## Actual full-size capacity

```powershell
.\.venv\Scripts\python.exe -m examples.enriched_capacity --capacity outputs/matrixfree-fullplate-capacity-20261107 --output outputs/enriched-capacity --max-seconds 90 --max-rss-mib 1024
```

Use a fresh output directory and the retained capacity inputs. The driver
checks the original mesh/plan hashes, current source, drivers and native
runtime. It constructs the actual native space and a separately identified
topology-order positive mass, then checks intact-box moments through degree
two. Native and topology coefficient permutations are not assumed equivalent.

The unchanged 500 x 500 x 3 mm, 73,487-tetrahedron mesh contains 391,207 scalar /
1,173,621 vector enriched coefficients. All nodal masses are positive and
total 2.025 kg. The ten mass-moment checks have maximum relative error
1.51e-13. Native/topology coefficient permutation remains unverified at this
size. No stiffness matrix, CUDA allocation or time step was constructed.

The run completed in 4.241 monitored seconds at 303.84 MiB sampled combined
RSS, 50.8% system RAM and at least 7,978.61 MiB available. System GPU readings
were 759 MiB / 31 C while computation used CPU; CPU temperature was unavailable.
The topology workspace screen is 89.69 MiB. A future dense local-stiffness
representation would require an estimated 1,135.34 MiB, not measured VRAM.
Raw evidence is under `outputs/enriched-fullplate-capacity-20261108`.

## Explicit time-step restriction

The conservative limits on these small controls are 80.80 ns rectangular
and 93.03 ns skewed. The existing Verlet integrator applies its 0.95 safety
factor. The implicit reference's 125 ns increment therefore cannot simply be
reused for this explicit method. The actual full-mesh stability limit remains
to be computed; positive mass alone does not establish time stability.

## Focused regression and retained evidence

All 29 selected enriched-element, explicit-integrator and implicit-integrator
checks pass with actual NGSolve, Gmsh and Elmer references enabled, without
skips. Test time is 21.345 seconds; monitored time is 22.515 seconds. Peak
sampled combined RSS is 312.02 MiB, system RAM 59.1%, available RAM at least
6,640.02 MiB and system GPU observations 2,218 MiB / 38 C. Computation used
CPU; CPU temperature was unavailable. The initial targeted run's four optional
Elmer skips are retained, followed by the fully enabled pass.

This isolated candidate adds six new checks. The last complete integrated
CPU/GPU run remains the earlier 253-test milestone; it was not repeated for
this separate element/driver addition. Current evidence is in
`evidence/enriched-tetrahedron-debug.json`, fingerprint
`762dbcfe826e28eff5be82ec17d8dc6bb3a9d41ebbd89a0b812dab0e6ba49eb2`.
It binds source/runtime, arrays, the original coarse temporal failure,
capacity receipts and focused tests. The November 7 usage-limit approval
failure prevented the pending launch; the final checks resumed November 8.
No full training ran.

## Remaining acceptance work

The next implementation must establish the full native/topology map, accurate
circular traction/probe integration for cubic surface traces, a scalable GPU
stiffness path and a mesh-specific stability limit. Full observation-window
cost, Lamb-mode dispersion, spatial/thickness/time refinement and representative
recess/frequency cases remain necessary. Coupled or calibrated PZT voltage and
physical agreement remain open. No accepted research bank or full training
result follows from these element/capacity checks.
