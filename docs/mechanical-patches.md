# Mesh-independent circular mechanical patches

The local elastic pilot applies unit-total-force circular loads and measures
the corresponding area-averaged displacement. Both use the same nodal weights,
so the pair remains mechanically reciprocal. It is a force/displacement
reference problem, not a bonded-PZT voltage or strain model.

Earlier smoke banks selected complete top triangles by centroid. Remeshing
therefore changed the effective footprint area and center. The current method
integrates each linear shape function over the actual triangle/disk intersection.
It preserves the requested circular area and force center on any valid top
triangulation, including coarse triangles that straddle the disk boundary.

## Integration

Translate/scale coordinates so the disk is centered at the origin with radius
one. [Green's theorem](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/3.-double-integrals-and-line-integrals-in-the-plane/part-c-greens-theorem/)
gives the area and first moments through boundary integrals:

```
A  = 1/2 integral (x dy - y dx)
Jx = 1/3 integral x (x dy - y dx)
Jy = 1/3 integral y (x dy - y dx)
```

Split each oriented triangle edge at its intersections with the unit circle.
An interior segment p→q contributes `cross(p,q)/2` to area and
`cross(p,q)*(p+q)/6` to the first-moment vector. Exterior pieces contribute
the corresponding oriented circle arc. For endpoint angles a→b the arc
contributes `(b-a)/2`, `(sin(b)-sin(a))/3`, and `(cos(a)-cos(b))/3`.
Radial edges cancel, leaving the intersection boundary. Vertex orientation
is accounted for explicitly. Small roundoff contributions are bounded and
near-zero negative nodal integrals are clipped before restoring the area sum.

Linear shape functions require only these three moments. Solve the affine
barycentric system for their integrals and multiply by radius squared to
recover m². Sum contributions into top-surface nodes, then normalize by the
actual patch area for the unit-force/average-displacement weights.

`surface_patch_with_report` retains actual area, first-moment centroid,
support-node and intersected-triangle counts, direction and edge clipping.
Bank generation requires full circular footprints and checks area/centroid
relative errors before solving. The standalone FEM helper supports clipped
edge footprints and reports their actual area/centroid instead.

Tests check full/half/quarter disks, triangles entirely inside/outside the disk,
tangencies, all vertex permutations, translated/scaled SI cases, affine
displacement averages and mesh independence. General nonsymmetric intersections
are also checked against independent adaptive vertical-slice quadrature.
Existing mechanical reciprocity and dynamics tests still apply.

This fixes footprint integration, not guided-wave resolution, adhesive
coupling, electrical loading or PZT physics. Displacement is still approximated
by the mesh's P1 basis; even an exact load integral cannot compensate for a
coarse bending field. Forward caches now include the patch integrator's source
hash. Historical banks retain their original hashes/provenance and are not
silently regenerated or promoted to research-ready data.
