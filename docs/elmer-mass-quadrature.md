# Corrected mass quadrature for the P2 reference

The first independent NGSolve comparison exposed an error in our Elmer
reference configuration. Earlier P2 operator and transient inputs used the
release's default quadrature. It integrates P2 elastic stiffness correctly
on straight tetrahedra, but does not integrate the quartic products in the
P2 mass matrix exactly. Those runs were provisional and retain their original
reports; they must not establish reference-waveform acceptance.

The packaged `elements.def` selects 27 points by default for type 510 and
64 points for its higher rule. The 27-point rule is a three-point tensor
Gauss rule mapped to the tetrahedron. The Duffy-map Jacobian raises the
polynomial degree, making that rule insufficient for all degree-four mass
products. Type 504 selects four default points and eleven higher points.
Both corrected rules integrate the required mass products for straight
geometry and constant density.

TRAXIS operator exports and native transients now explicitly set
`Relative Integration Order = Integer 1`. Their integration choices match;
the native/implicit comparison cannot silently use different masses. No
vendor file or global installation was modified. The waveform cache also
binds `elements.def`, whose contents control element integration defaults.
Source hashes distinguish all regenerated groups from older results.

## Independent evidence

For the ten-node tetrahedron, an exact reference expands each quadratic
shape-product in barycentric monomials and integrates
`integral(product(lambda_i^a_i)) = 6*V*product(a_i!)/(sum(a_i)+3)!`.
This uses no numerical quadrature or external solver. An independent
four-point-per-axis Duffy/Gauss integration agrees with that reference;
the three-point-per-axis rule reproduces the integration defect.

On rectangular and affine-skewed blocks with 24 tetrahedra and 75 P2 nodes,
the original Elmer mass differs from the exact nodal matrix by **4.656%**
relative Frobenius norm. NGSolve's transformed mass differs by at most
2.55e-14. Elmer's higher rule reduces the rectangular mass error to
**1.45e-15** in the isolated integration-order probe. Stiffness already
agrees between the libraries to about 2.4e-14.

The regression tests check exact moments, positive mass, node/midpoint
permutations, rejected curved geometry, and both native integration choices.
The lower-order rule must exhibit a measurable error; the production rule
must match the exact mass below 1e-10. Native transient comparisons also
pass with the corrected export and time-domain settings.

Relevant pinned implementations are
[StressSolve.F90](https://github.com/ElmerCSC/elmerfem/blob/5a8de867068be0568f09af40fb90fee300dfbede/fem/src/modules/StressSolve.F90#L1110),
[Stress.F90](https://github.com/ElmerCSC/elmerfem/blob/5a8de867068be0568f09af40fb90fee300dfbede/fem/src/Stress.F90#L196)
and [Integration.F90](https://github.com/ElmerCSC/elmerfem/blob/5a8de867068be0568f09af40fb90fee300dfbede/fem/src/Integration.F90#L1829).

## Interpretation of older results

The P2 dispersion, transient, aperture and finite-plate results published
earlier in this repository used the previous integration setting. Their
algebraic comparisons established agreement for that discrete system, not
exact continuum mass integration. Their failed spatial screens and
`ready_for_training_data=false` flags remain unchanged. Their numerical
values describe the historical inputs and require regeneration before
being cited as results of the corrected formulation.

This fix does not establish spatial convergence, a PZT voltage response or
physical accuracy. Those remain separate requirements. The independent
[high-order assembly check](highorder-reference.md) verifies corrected
P1/P2 operators and exact represented fields through degree four.
