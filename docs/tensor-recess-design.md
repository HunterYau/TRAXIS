# Conforming tensor meshes for backside recesses

This remains the mapped recess solver design, not an accepted wave result.
The frozen intact tensor study has completed with its original sources.
Explicit conforming material geometry now passes independent native checks;
see conforming-recess-geometry.md and recess-shape-refinement.md. Mapped
elastic stiffness, positive mass and conservative stability bounds now pass
small independent native checks; see mapped-recess-operators.md. Mapped GPU
operators, physical disk ports and complete small pulse controls now pass
checks; see mapped-recess-gpu.md and mapped-surface-ports.md. Full-size compact
capacity is measured in full-size-recess-capacity.md. The existing CAD
tetrahedral references remain separate evidence.

The current target is the user's revised 500-by-500-by-3.175 mm plate with flat-bottom
circular backside recesses, diameters 20/30/40 mm and depths
0.75/1.50/2.00 mm. The upper surface stays flat. A depth map label does not
define the three-dimensional elastic domain by itself.

## Geometry and topology

The current `TensorLayout` has one affine Jacobian and implicit Cartesian
connectivity. Changing a local thickness value in that class would not
produce a conforming recess wall. The extension needs explicit shared-node
connectivity and element/quadrature-dependent geometry.

For the first controlled circular recess, use a block decomposition of the
planar domain: a central quadrilateral core, curved blocks around the circle,
and surrounding blocks reaching the square plate boundary. Extrude matched
surface meshes through the recess-floor plane. Below that plane, omit the
disk volume; above it, retain the remaining stock. A paired intact control
fills the lower disk. Shared interfaces must identify the same nodes, with
no duplicate faces or unverified hanging-node constraints.

Gmsh's documented transfinite and extrusion methods support structured
quadrilateral/hexahedral construction. This supports a block-mesh prototype;
it does not guarantee an all-hexahedral mesh of an arbitrary CAD subtraction.
See the official [transfinite tutorial](https://gmsh.info/doc/texinfo/#t6)
and [mesh-module description](https://gmsh.info/doc/texinfo/#Mesh-module).
The installed Gmsh version inspected on November 10 is 4.15.2. No package
change was made.

Keep the represented geometry fixed while refining the displacement space.
Check circular-wall radius, recess area, volume and remaining thickness
against the analytic CAD dimensions, including between geometry nodes.
Vary geometry order/resolution separately to measure the shape error; a
polynomial boundary through exact circle nodes is not automatically an
exact circle. Reject negative or poorly conditioned geometric Jacobians.
The analytic removed volume is pi * radius^2 * depth.

## Operator and mass

Use the same checked strain/stress contractions, now with the mapped inverse
Jacobian and positive determinant at each volume quadrature point. Assemble
positive GLL mass from these physical volume weights. Retain local sum
factorization rather than storing dense element stiffness matrices. Screen
explicit connectivity, geometry and integration buffers before allocation;
the existing Cartesian storage forecast does not cover these new arrays.

Derive and independently check a stability bound for the mapped element
family. The intact plate's bound cannot be reused in a thin remaining-stock
region. Freeze a common step for each paired/refinement experiment after
measuring actual geometry and operator capacity. Measure full-window cost
under unchanged RAM/VRAM/temperature guards before planning banks.

## Independent reference and physical ports

The installed NGSolve 6.2.2606 API exposes `Mesh.SetDeformation(GridFunction)`
and `Mesh.Curve(order)`. Official [linear-elasticity documentation](https://docu.ngsolve.org/ngs24/SaS/linearelasticity.html)
uses curved CAD meshes. A small mapped-hexahedron reference can use a fixed
polynomial deformation and native assembly to compare the same represented
geometry. This is a proposed reference path; API availability is not a
successful mapped-element comparison by itself. A subsequent bounded
nonaffine Q3/Q4/Q5 reference now passes native operator, geometric-volume,
affine-energy and six-rigid-mode checks; see mapped-tensor-reference.md.
It uses a manufactured polynomial map, not the circular-recess topology.

Before a recess wave run, compare every operator column and mapped mass on
bounded connected fixtures, including curved/skewed elements. Check shared
continuity, affine strain energy, six rigid motions, geometry volume/moments,
positive mass and an independently computed spectrum/stability bound. Keep
exact-integration and GLL-integration comparisons explicitly distinguished.

The mechanical ports remain physical-area integrals with analytic disk-area
normalization, signed weights and work-conjugate probes. A separately checked
mapped routine now integrates circular disks on the ruled curved top faces.
Native shapes and independent Gaussian quadrature verify complete covectors;
represented physical moments and actual per-state mounting geometry are
checked. This does not identify the real clamp/PZT transfer function.

## Acceptance sequence

First verify small mapped operators and geometry; then paired intact/recess
small waves and separate geometry/space/time comparisons; then actual
full-size capacity and complete windows. Cover the planned defect dimensions
and locally thin regions without tuning criteria to the observed outcome.
Preserve raw failures and unchanged physical source/receiver footprints.

Even a passing mechanical recess study does not identify removable-clamp
preload/contact, PZT traction or electronics response. The synthetic
electrode/circuit reference, actual part/layout decisions and physical
calibration remain necessary.
Only accepted bank generation may feed the later matched reconstruction,
sensing-policy, resolution and uncertainty experiments. No training or
physical validation is implied by this design note.
