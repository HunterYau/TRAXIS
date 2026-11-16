# Repeated geometry basis evaluation

CPU profiling of a 96,627-DOF coupled-layer operator attributed 1.040 of
1.580 profiled seconds to 1,344 calls rebuilding the same Lagrange values
and derivatives. The cache retains at most sixteen reference grids of at
most 1,024 points. It keys exact float64 point bytes and degree, preserves
the original Polynomial evaluation, and returns fresh writable arrays in
their original layout. Large grids use the original calculation directly.

Verification used the actual uncached source from a retained wave snapshot
(SHA256 de20a3763cbbec9c22cfc1aabfd3c5afc41af45da14054a42ad28fdbd7c6f9bf).
Fifty basis arrays and six intact/recessed Q3/Q4/Q5 coupled-layer mesh/operator
assemblies match bit for bit. All 23 focused geometry/operator tests passed
with no skips, including the independent native geometry, all-column matrix
and spectrum comparison.

The complete 12,431,655-DOF recessed CPU preflight was also repeated. Every
retained point, connectivity, mass, inverse-Jacobian and volume-weight hash
matches the original; its stability bound is exactly unchanged. Observed
setup time fell from 261.477 to 115.720 seconds. The cached run peaked at
1,048.65 MiB RSS and 67.5% system RAM. These observations came from runs
alongside other GPU work; they are not a controlled benchmark or a wave
trajectory speedup claim.

The reviewed source transition and historical scattering assessments were
repeated with the cache source. The CUDA integrator comparison still has
zero differences. Temporal scattering still passes; spatial scattering
still fails at receiver 3 with a 1.344761% finest change. Original run and
CPU proof files were retained unchanged. No convergence threshold changed.

Compact receipts are in `evidence/geometry-basis-cache-20261115.json` and
`evidence/geometry-cache-archive-review-20261115.json`. This branch remains
separate from the source of the live in-plane run until that run and its
independent CPU check finish. This work does not establish full 3D spatial
convergence, a calibrated PZT response or readiness for training data.
