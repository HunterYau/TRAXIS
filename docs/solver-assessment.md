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
| Pogo | Promising alternative for GPU explicit linear elastodynamics and large waveform banks | The original paper establishes the elastic solver; current distribution, maintenance, Windows support, and bonded-PZT workflow have not been verified here |
| Ansys coupled-field transient piezoelectric analysis | Credible alternative if already available through the school | Solver choice, mesh and performance need a project-specific pilot; do not assume every Ansys product includes the same features |
| OnScale | Documented time-domain electromechanical transducer modeling | Sources describe capability, not confirmed current account availability or price |
| Local linear-tetrahedron FEM | Transparent regression reference; traction-free recesses and finite boundaries; optional CAD-conforming mesh; no extra wave-solver license | Low-order elements need fine meshes; no adhesive, PZT domain, electrodes, electrical loading, or voltage readout |
| Elmer FEM | Free independent elastic reference; portable Windows release and tiny 3D stiffness/consistent-mass checks actually tested | Full-plate transient/high-order cost, spatial convergence and bonded-PZT voltage workflow remain unverified |
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

## Primary sources

- [COMSOL 6.4 Elastic Waves, Time Explicit](https://doc.comsol.com/6.4/doc/com.comsol.help.aco/aco_ug_elastic_waves.06.30.html)
- [COMSOL 6.4 Lamb-wave NDT plate example](https://doc.comsol.com/6.4/doc/com.comsol.help.models.aco.lamb_waves_ndt_plate/lamb_waves_ndt_plate.html)
- [COMSOL transient piezoelectric formulation](https://www.comsol.com/blogs/modeling-piezoelectricity-using-the-discontinuous-galerkin-method)
- [Huthwaite (2014), GPU explicit elastodynamics and Pogo](https://doi.org/10.1016/j.jcp.2013.10.017)
- [Ansys piezoelectric analysis documentation](https://ansyshelp.ansys.com/public/views/secured/corp/v252/en/ans_cou/Hlp_G_COU3_piezo.html)
- [OnScale full 3D transducer example](https://support.onscale.com/hc/en-us/articles/360009768292-Simulating-Full-3D-Transducer)

