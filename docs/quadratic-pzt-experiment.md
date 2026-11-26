# Experimental quadratic PZT tetrahedra

Branch: `research/pzt-quadratic-tetrahedra`. The parent is the existing PZT
mesh study; its pending restored 5.340576171875/10.68115234375 ns settings
were copied into this branch without changing the original study worktree.
No time stepping occurs in the experiments below.

## Implementation and fallback

`traxis/quadratic_piezoelectric.py` is an optional, separate implementation.
The original linear body, physics assembly, LTS recurrence, GPU kernels,
CUDA graph backend and scheduler defaults were not edited. SHA-256
verification covers all 692 inherited files and the original study checkout.
See `evidence/quadratic-pzt-isolated/protected-source-verification.json`.

The new elements use the standard ten-node P2 Lagrange basis for BOTH
displacement and electrical potential. Shared global edges have shared
midpoints. Geometry is straight-sided and affine, with the original polygon
boundary preserved; curved midpoint coordinates are rejected. Changing the
angular count changes polygon geometry, and is reported as such.

Stiffness, dielectric and piezoelectric coupling integrands have degree two
and use a degree-two-exact four-point tetrahedron rule. Consistent mass has
degree four and uses positive Duffy/Gauss quadrature exact through degree
four. Mass is never lumped: P2 corner shape integrals can be negative and
row-sum lumping would be inappropriate. Sparse component assembly supports
small frequency-domain experiments. `attach_quadratic_disk` is an opt-in
perfect-tie adapter returning the existing BodyAttachment contract, with all
interface midside DOFs included and both mechanical/electrical backreaction.
Dense attachment conversion has explicit memory checks. Elastic glue and
curved geometry are unsupported by this experimental adapter.

The ORIGINAL linear implementation remains the default/fallback. Nothing
automatically changes the scheduled simulations to quadratic elements.

Theory sources:

- [Abaqus interpolation definitions](https://docs.software.vt.edu/abaqusv2026/English/SIMACAETHERefMap/simathe-c-tritetwedge.htm)
- [DefElement Lagrange element definition](https://defelement.org/elements/lagrange.html)

The edge ordering is explicitly declared in the module; it need not match
an external package's numbering. Full consistent mass and adequate positive
quadrature are verified independently against higher quadrature orders.

## Small tests and adaptive investigation

Six tests check polynomial reproduction and gradients; bending/rigid modes;
consistent mass positivity and quadrature; electrode potential/capacitance;
midpoint sharing; and two-sided plate attachment through the original mass
merging backend. The controlled quadratic bending error is at roundoff.

The initial cost/accuracy guess was radial/angular/thickness **3/32/2**.
A finer **4/32/3** reference produced nearly identical isolated radial
resonance and six-frequency voltages. Cheaper meshes were then tried,
followed by angular refinement. Thirteen isolated free-disk experiments
were completed, each with a radial modal calculation and six harmonic
responses. Electrical and mechanical material values stayed fixed.

Radial response alone was insufficient: subsequent receiver tests prescribed
analytical S0 and A0 surface displacements at the bottom PZT face using
`fundamental_profile`, with a fixed 1 nm displacement scale, a 1 mm aluminum
plate profile, E=71 GPa, nu=.33 and rho=2700 kg/m3. There is NO plate transient
or simulated plate backreaction in this diagnostic. Complex voltages compare
both amplitude and phase, without gain, phase, or delay fitting. For both
modes all six frequencies 50,100,150,200,250,300 kHz were used.

The cheap meshes underperformed in A0, so radial and thickness refinement
continued. Ten prescribed-base configurations completed; the next 7/64/2
configuration was recorded but not assembled because it exceeded this
study's conservative 8192-node / 2.5 GiB forecast budget. Its forecast was
about 2817 MiB with about 5040 MiB available. These are experimental limits,
NOT a claim that it physically cannot fit. Largest measured component RSS
was below 900 MiB across the tests. Large allocations are checked against
current available RAM; no GPU or RunPod was used.

The final radial comparison, **5/64/2 vs 6/64/2**, gave six-frequency complex
voltage L2 differences **S0 0.2807%, A0 0.8052%**. Its worst individual A0
difference was **4.788% at 150 kHz**, where cancellation makes the net voltage
small. At 200 kHz it was 2.233%. Thus this candidate meets an exploratory 1%
aggregate-response criterion, but NOT a 1% all-frequency criterion. No mesh
is declared fully converged to 1% or 0.1%, optimal, or validated for the plate.

Independent angular and thickness checks are reported separately; neither
the finest mesh nor an aggregate score is treated as exact ground truth.
The finer mesh comparisons are useful evidence, not true-solution errors.
Tests suggest quadratic interpolation fixes the identified polynomial
bending weakness while general A0 receiver convergence still needs care.

## Saved results and commands

`evidence/quadratic-pzt-isolated/results.json` retains all tested configurations,
per-frequency voltages, errors, eigen residuals, time and sampled RAM.
`results.txt` and `frequency-errors.png` provide a readable summary.
Timing occurred under mixed concurrent CPU load and is exploratory; it does
not predict a full plate GPU speedup. Small sparse receiver memory also
does not estimate the full dense attachment/GPU footprint. The local RTX
3060 Ti has 8 GB VRAM and the host about 16 GB RAM; a full plate calculation
requires separate resource and time-step validation.

From this experimental worktree, with a normal activated Python environment:

```powershell
python -m unittest tests.test_quadratic_piezoelectric
python -m examples.quadratic_pzt_benchmark --rings 3 --sectors 32 --layers 2 --output outputs/quadratic-repeat/radial
python -m examples.quadratic_pzt_basewaves --rings 5 --sectors 64 --layers 2 --output outputs/quadratic-repeat/basewaves
python -m examples.quadratic_pzt_report
```

These commands perform isolated component tests only. Use fresh result
folders when changing the implementation; completed experiments are never
overwritten. The reporter reads the saved study folder and does not start
calculations. Archived initial radial receipts predate source-fingerprint
caching and will require a fresh output folder for a repeat.
