# Forward-solver assessment

Assessment started 2026-11-03; Elmer reference trial added 2026-11-04.
This is an engineering recommendation based on
documented capabilities, not a measured head-to-head solver benchmark.
No commercial solver has been installed, licensed, or run for this project.

A read-only check of the standard 32/64-bit Windows uninstall registries on
this PC found no COMSOL, Ansys, Abaqus or OnScale entries. This does not rule
out portable installations, a school license, or a remote machine. Reference
solver access remains unconfirmed; the software workflow can proceed while
that is resolved.

## Recommendation

Use COMSOL Multiphysics as the reference if the necessary modules and license
are available. The project needs 3D dispersive guided elastic waves in a
finite plate, backside recess geometry, reflections, and eventually bonded
PZT transmit/receive coupling. COMSOL documents both a 3D Lamb-wave NDT example
and a coupled transient piezoelectric interface. Build our own Python
experiment framework rather than a full production multiphysics package.

| Candidate | Fit | Important limit |
| --- | --- | --- |
| COMSOL Elastic Waves / Piezoelectric Waves, Time Explicit | Preferred reference; explicit high-order elastic propagation and electrostatic coupling; documented plate NDT example | Commercial access and exact module entitlement must be checked; full-plate cost remains to be measured |
| Pogo | GPU elastodynamic solver; Imperial documents NVIDIA hardware and Windows/Linux support | Current installer/license access, project-specific performance and bonded-PZT voltage workflow remain unverified |
| Ansys coupled-field transient piezoelectric analysis | Credible alternative if already available through the school | Solver choice, mesh and performance need a project-specific pilot; do not assume every Ansys product includes the same features |
| OnScale | Documented time-domain electromechanical transducer modeling | Sources describe capability, not confirmed current account availability or price |
| Local linear-tetrahedron FEM | Transparent regression reference; traction-free recesses and finite boundaries; optional CAD-conforming mesh; no extra wave-solver license | Low-order elements need fine meshes; no adhesive, PZT domain, electrodes, electrical loading, or voltage readout |
| Elmer FEM | Free independent elastic reference; portable Windows release and tiny 3D stiffness/consistent-mass checks actually tested | Full-plate transient/high-order cost, spatial convergence and bonded-PZT voltage workflow remain unverified |
| NGSolve | P1-P4 mechanical checks, corrected native P2 agreement, and passing small intact/paired-recess discrepancy screens | Installed Windows build has CUDA disabled; full-size capacity, PZT voltage coupling and physical calibration remain unverified |
| Scalar acoustic / straight-ray synthetic data | Useful for plumbing and deliberate mismatch experiments only | Cannot establish quantitative thickness reconstruction for dispersive elastic plate waves |

The Pogo paper reports large speedups in its own benchmark; these are not
predictions for this computer. A GPU is not evidence of sufficient VRAM.

## Reference COMSOL pilot specification

1. Use SI units internally. Plate: 0.500 x 0.500 x 0.003 m, with recorded
   6061-T6 density, Young's modulus and Poisson ratio. Initial nominal values
   in Python are assumptions, not measured coupon properties.
2. Keep finite free edges if the real edges are free. Represent measured
   supports explicitly. Absorbing boundaries are a separate experiment and
   must not silently remove real reflections.
3. First compare intact elastic propagation with a known finite-area
   mechanical source. Then add measured PZT/adhesive geometry, poling,
   piezoelectric constants, electrodes, drive impedance and receiver loading.
   Mechanical displacement or velocity is not PZT voltage.
4. Model backside flat-bottom recesses by subtracting material. Candidate
   diameters are 20/30/40 mm; depths 0.75/1.50/2.00 mm. Keep at least 1 mm
   of remaining stock for the deepest planned recess. Mesh the floor, rim,
   thickness and transducer interfaces, not only the plate outline.
5. Determine the shortest relevant guided-mode wavelength over the actual
   pulse bandwidth, including locally thinned plate regions. Do not use
   bulk longitudinal speed alone to choose mesh size. The high-order dG
   recommendations in COMSOL do not apply to our linear tetrahedra.
6. Run at least three spatial refinements and independently refine time
   sampling/internal time step. Compare arrival times, amplitude, phase,
   envelope, and reflected packets on aligned physical probes. Record
   wall time, peak RAM/VRAM, degrees of freedom and mesh quality.
7. Compare mode dispersion to an independent Rayleigh-Lamb solution and
   intact physical traces before building thousands of virtual plates.
   Calibrate on the control/calibration plates only; freeze settings before
   using final holdouts. Choose numeric tolerances from measurement
   repeatability, not after seeing the final test errors.
8. Save each transmitter-frequency solve once with all receiving traces.
   One revealed receiver record is the plan's policy budget unit; an actual
   excitation may acquire many receivers. Report both costs for real trials.

## Local numerical formulation

The local backend solves `M u'' + K u = f` for small-strain isotropic elasticity
with linear tetrahedra and row-sum lumped mass. Natural boundaries are free
surfaces. No scalar-wave or bulk-speed approximation substitutes for the
elastic stiffness. The structured backend excludes complete cells; the optional
Gmsh/OpenCASCADE backend meshes a CAD-subtracted circular recess. Both require
reported geometric error and independent wave convergence. The CAD pilot reduces
one recess-volume error from 49.2% to 0.42%; this does not validate PZT waveforms.

Time integration uses velocity Verlet (central differences). A conservative
bound on the largest eigenvalue of `M^-1/2 K M^-1/2` gives the explicit step
limit. Diagnostic discrete energy is checked after forcing stops. Spatial
convergence, dispersion, source/receiver fidelity and intact physical
agreement remain separate requirements even when algebraic tests pass.

A bounded intact-plate sweep on 2026-11-04 confirmed this distinction. With
identical source/probe footprints and a common fine time step, the final mesh
refinement changed some raw traces by 146%. Independent temporal refinements
on the finest mesh showed order 2.00025 and a maximum final change of 0.0209%.
HXT/Netgen improved a sliver-driven stability bottleneck but did not establish
spatial accuracy. See [waveform-validation.md](waveform-validation.md). These
local banks must not be used as research training data; no commercial/physical
reference comparison has yet been completed.

The optional [Elmer reference workflow](elmer-reference.md) now verifies the
local stiffness and lumped-mass assembly against an independent solver on
rectangular and skewed 3D tetrahedral meshes, with relative matrix errors
below 1.4e-14. The portable release runs inside this project without global
installation. This removes commercial access as a prerequisite for beginning
independent mechanical checks. It does not validate coarse spatial resolution
or establish a production PZT solver choice; COMSOL remains the preferred
coupled-wave reference if accessible, pending a measured project benchmark.

An additional [quadratic Elmer dispersion study](elmer-waveguide.md) uses
independent P2 operators at 100/200 kHz in 1/3 mm stock. Its maximum finest
analytic frequency error is 0.0677%, but one strict thickness-refinement
decrease criterion failed. These promising bounded infinite-plate results
do not establish finite-plate waveform, recess-scattering or PZT accuracy.

The [Elmer transient interface checks](elmer-transient.md) also pass for
uniform volume/nodal force equivalence and a manufactured discrete elastic
mode. Both P1/P2 systems show second-order time convergence. This verifies
the chosen load/probe/integration interface, with spatial convergence and
transducer coupling still required before accepting reference waveforms.

The [bounded P2 waveform path](elmer-waveforms.md) now exports independent
Elmer operators once and integrates them in a monitored sparse worker.
The small native-transient comparison agrees to 1.74e-12 relative L2, and
a 3,777-node finite plate runs in 9.69 seconds at 302.2 MiB sampled combined
RSS. Native bulk-matrix reuse crashed in isolated tests on this release;
the worker avoids that path. This is a practical mechanical reference
candidate, pending finite-plate/recess refinement and physical PZT validation.

The first [P2 finite-plate refinement](elmer-plate-refinement.md) has now
completed. Its temporal screen passed at 0.2205% maximum change and order
1.99917; its spatial screen failed at 30.84% between the finest two meshes.
The 15,478-node reference fits a 768 MiB guard after sparse memory fixes,
but these discretizations cannot yet supply research-quality plate traces.

An [independent directional P2 study](elmer-directional-refinement.md) also
fails its in-plane and thickness screens at 12.83% and 8.01%, respectively.
Time refinement passes at 0.2205% with order 1.99909. The fine intact grid
uses 15,129 nodes and about 1,010 MiB sampled outer process-tree RSS. The
separate axes support further spatial-resolution work, rather than accepting
the current mechanical references for research data.

The [independent high-order assembly check](highorder-reference.md) exposed
under-integration in our earlier Elmer P2 mass configuration. Exports and
native transients now explicitly select the higher rule and pass an exact
quadratic mass reference. Earlier P2 numbers on this page are historical
and need regeneration for the corrected formulation; see the
[quadrature correction](elmer-mass-quadrature.md). This fixes an assembly
configuration error without establishing wave/PZT/physical acceptance.

## Full-size decision update, November 7, 2026

The [paired CAD control](highorder-recesses.md) passes its declared current
and current-minus-intact mesh/time screens on a 20 mm plate. A separate
500 mm geometry-only screen rejects the current local element/assembly
budget before creating a mesh. This establishes a limit of the bounded worker
configuration, not an intrinsic inability of NGSolve to model the full plate.
No full-size performance number or accepted training bank is available.

A guarded inspection of this installed NGSolve 6.2.2606 build reports
`USE_CUDA=False`. A GPU path must not be advertised for this environment.
NGSolve's official [matrix-free tutorial](https://docu.ngsolve.org/latest/i-tutorials/unit-2.11-matrixfree/matrixfree.html)
and [explicit DG wave example](https://docu.ngsolve.org/latest/i-tutorials/unit-5.5-cuda/wave_cuda.html)
document candidates for reducing matrix storage. The latter is a scalar
first-order wave example, not a validated elastic plate/PZT implementation;
it cannot substitute for vector elasticity. The current online tutorial
version also differs from the pinned local build. Any local extension must
retain exact-field, traction, mode-dispersion, mesh/time and resource checks
before comparing physical plate results.

COMSOL remains the preferred coupled reference if accessible. Its
[documented Piezoelectric Waves interface](https://doc.comsol.com/6.4/doc/com.comsol.help.aco/aco_ug_intro.04.14.html)
couples transient elasticity and electrostatics. This is capability evidence,
not a measured project benchmark or confirmed module entitlement. GPU support
documented for pressure acoustics must not be assumed to cover this coupled
elastic application. No license purchase, installation or commercial solve
has occurred in this work.

[Imperial's Pogo project page](https://www.imperial.ac.uk/non-destructive-evaluation/publications-products-and-services/pogo/)
confirms its elastodynamic focus and Windows/Linux NVIDIA support. The linked
vendor site returned HTTP 403 during this check, so current download/license
availability and an actual PZT workflow remain unverified. No vendor speedup
claim is used as a prediction for this machine.

The next measured comparison should preserve the intended 500 x 500 x 3 mm
plate, physical observation duration, source/probe definitions and defect
dimensions. Compare a scalable elastic implementation or accessible external
solver against the retained small references; then establish PZT/adhesive,
electrode, drive and receiver-loading inputs and intact physical agreement.
Preserve vector elasticity and the acquisition window needed for relevant
arrivals, and accept research banks only after those checks pass.

## Measured matrix-free capacity, November 7

The [matrix-free H1 candidate](matrixfree-reference.md) now constructs the
actual 500 x 500 x 3 mm intact P3 mesh/operators with 1,319,508 vector
coefficients, without global sparse assembly. It passes translation mass,
rigid null-force and work reciprocity in 64.15 monitored seconds at 482.6 MiB
sampled combined RSS. This supersedes the earlier preflight-only limit as
evidence about this specific intact matrix-free allocation path; it does not
retroactively turn the earlier recess/assembled request into a successful run.

Two full-size stiffness actions cost about 6.5 seconds each and mass actions
2.7 seconds. Iterative transient solves need many such actions, so fitting in
RAM does not establish practical full-window propagation. The next benchmark
must measure an actual acceleration solve at the intended time step and
source, including strict residual/iteration costs, without accepting truncated
arrivals as a research window. Efficient preconditioning or a different
verified elastic implementation may still be necessary.

All 239 native-enabled regression tests pass without skips; tiny all-column
P1-P4 comparisons and direct/iterative transients agree, with checked zero-new-
solve recovery. See evidence/matrixfree-debug.json for frozen sources, arrays,
failed attempts and resource receipts. Coupled/calibrated PZT voltage and
physical agreement remain open. The installed build still has CUDA disabled;
no full training or accepted full-size time propagation has run.

The [actual first-step probe](matrixfree-step-cost.md) now measures this
limitation directly: the unchanged full-size P3 mesh, 125 ns step and checked
3 mm-radius force disk reach the 180-second wall guard with fourteen completed
CG iterations and no independently accepted state. Peak sampled combined RSS
was 494.48 MiB. Iterations cost roughly 9-10 seconds; no final true residual is
available. This cost stop does not establish a numerical-accuracy failure.
It does rule out treating the memory-capacity result as proof of an affordable
research-data workflow. All 243 native-enabled regression tests pass; the
small first-step control independently agrees with sparse Newmark.

A faster backend must preserve the full vector elastic and consistent-mass
operators and repeat the same independent checks. CuPy's official
[installation documentation](https://docs.cupy.dev/en/stable/install.html)
describes Windows CUDA wheels and optional CUDA component wheels that avoid
a system-wide toolkit installation. This is a possible route for a separately
bounded GPU element-operator candidate, not an implemented or verified TRAXIS
GPU solver. No package/toolkit/driver change or GPU computation occurred in
this cost investigation. Scalar acoustics must not replace elastic plate waves.

## Primary sources

- [COMSOL 6.4 Elastic Waves, Time Explicit](https://doc.comsol.com/6.4/doc/com.comsol.help.aco/aco_ug_elastic_waves.06.30.html)
- [COMSOL 6.4 Lamb-wave NDT plate example](https://doc.comsol.com/6.4/doc/com.comsol.help.models.aco.lamb_waves_ndt_plate/lamb_waves_ndt_plate.html)
- [COMSOL transient piezoelectric formulation](https://www.comsol.com/blogs/modeling-piezoelectricity-using-the-discontinuous-galerkin-method)
- [Huthwaite (2014), GPU explicit elastodynamics and Pogo](https://doi.org/10.1016/j.jcp.2013.10.017)
- [Ansys piezoelectric analysis documentation](https://ansyshelp.ansys.com/public/views/secured/corp/v252/en/ans_cou/Hlp_G_COU3_piezo.html)
- [OnScale full 3D transducer example](https://support.onscale.com/hc/en-us/articles/360009768292-Simulating-Full-3D-Transducer)

