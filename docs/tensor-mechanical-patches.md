# Tensor mechanical footprints and fifth-degree control

The tensor reference and CUDA action now accept Q5 in addition to Q1-Q4.
The CUDA shared arrays are sized at compilation for the local node count;
Q5 uses 256 threads for 216 scalar nodes. Native connected-element checks
cover every Q5 stiffness column, diagonal mass, exact stiffness and consistent
mass on a skewed fixture. This extends the operator reference, not the scope
of the previous full-size Q4 capacity measurement.

`traxis/tensor_patches.py` integrates the complete physical circular disk
against the piecewise polynomial top trace. It uses a polynomial
antiderivative in one coordinate and adaptive integration in the other,
splitting intervals at circle/face intersections. It normalizes by analytic
disk area, checks represented polynomial moments and retains signed weights.
The same sparse covector gives unit-total-force loading and work-conjugate
area-average displacement. It does not represent a PZT voltage response.
Only interior disks on an axis-aligned top surface are supported; clipped
disks, skewed physical footprints and excessive face counts are rejected.

`examples/tensor_patch_reference.py` independently integrates NGSolve shapes
in physical coordinates with Gaussian inner quadrature. Five Q3/Q4/Q5 cases
cover centered/offset disks, cell-boundary crossings and a single-cell disk.
The retained comparison is below 1.14e-12 relative against a fixed 1e-9
criterion. Analytic area and represented moment checks also pass.

The initial receipts are `outputs/tensor-q5-{pure,native}-tests-20261109`
and `outputs/tensor-circle{-native,}-tests-20261109`. They include invocation,
test logs, process-tree monitoring and raw native matrices/covectors. These
focused groups overlap earlier tests; the last full integrated suite remains
268 tests. Current-source verification and the subsequent full-window
measurements must retain their own receipts.

`configs/tensor-wave-protocol.json` specifies a proposed 500-microsecond
intact-plate comparison: fixed 150-by-150-by-1 cells, Q3/Q4/Q5 spatial degrees
at 65,536 steps and Q4 temporal levels at 32,768/65,536/131,072 steps. The
source and three receivers retain 3 mm radius disks and the existing normal
mechanical force/displacement convention. A configuration file alone is not
a completed or accepted waveform study. The earlier failed tetrahedral
thickness screen, defect geometry, calibrated PZT response and physical
agreement remain open. No research bank or training is accepted here.
