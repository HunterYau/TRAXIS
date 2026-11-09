# Factorized enriched GPU elasticity

The 15-node positive-mass candidate now has a float64 GPU stiffness action and
device-resident velocity Verlet. It retains the vector elastic weak form and
all enriched coefficients. The shared reference-gradient Gram factor has rank
13; this removes redundant gradient data, not displacement modes. Each element
stores geometry and fifteen scalar indices rather than a dense 45-by-45 matrix.

`gpu_enriched.py` checks every native element's nodal permutation, shared
entities, component ranges, positive mass and coordinates. Its maximum local
mass-normalized Gershgorin bound conservatively limits the assembled explicit
step. The integrator applies an additional 0.95 safety factor. Circular
footprints use cubic surface traces despite the enriched space advertising
order two. A quadratic-only trace request is rejected by the retained control.

`gpu_enriched_verlet.py` keeps states, stiffness actions, mass division and
probe accumulation on the GPU. No coefficient vector moves to the host per
step. Scalar energy/resource observations still synchronize. Probe compression
drops no nonzero coefficients. Requested intervals are never shortened to fit
a guard. Inputs, sample storage, vector workspace and free VRAM are screened.

## Retained small references

`outputs/enriched-gpu-corrected-reference-20261108` checks every column on two
six-tetrahedron controls with 153 vector coefficients each. Operator discrepancies
are about 1.2e-15; positive masses and native/topology maps agree. GPU/CPU Verlet
traces agree at 64/128/256 steps over each fixed manufactured-mode interval.
The rectangular driven disk experiment covers 20 microseconds at 200 kHz,
including reciprocity and postburst modified-energy checks. It is not a
full-plate acquisition. Only the rectangular case runs the driven patch checks.

The earlier `enriched-gpu-reference-20261108` report overwrote its stiffness
error variable with the final unforced waveform error. Its saved raw matrices
independently pass the intended comparison, but its mislabeled field and gate
must not be used as that proof. The corrected driver keeps distinct variables;
the test recomputes the reported stiffness norm from the saved matrices.
Unperformed driven checks now use null values and an explicit status flag.

All 17 selected enriched GPU/element/circular-patch tests pass without skips in
23.454 test seconds. The final standalone reference also passes and retains
plans, source/runtime/package bindings, arrays and monitor receipts. The last
complete integrated suite remains the earlier 253-test result; overlapping
selected suites are not added to that total.

## Full-size operator measurement

`examples.enriched_gpu_wave` binds the retained capacity mesh, source, driver,
native/GPU packages, explicit settings and acceptance criteria. Operator mode
does not propagate; wave mode must finish its entire declared window. Both
compare a nonaffine physical-coordinate GPU field action with independent
streamed exact local CPU matrices. The native-order positive mass is checked
against all intact-box moments through degree two. Circular patches check
their area and represented quadratic moments before propagation.

The actual unchanged 500 x 500 x 3 mm mesh has 73,487 tetrahedra and 1,173,621
vector enriched coefficients. Every element's native/topology map passes.
The independent full-size action discrepancy is 4.44948e-15 relative L2.
The conservative explicit limit is 19.8610 ns. At CFL fraction 0.8, the declared
500-microsecond window requires 31,469 steps of 15.8887 ns. Twenty repeated
device actions average 17.8820 ms wall / 17.8489 ms device time, versus 29.5028
ms for the host-facing action. The independent CPU action takes 2.59161 seconds.
No full-window throughput follows from this isolated action measurement.

Raw evidence is in `outputs/enriched-gpu-fullplate-operator-20261108`, which
completed in 48.3122 monitored seconds. The two preflight tests pass and keep
the observation window, stability/accuracy criteria, geometry and GPU limits
explicit. Settings are in `configs/enriched-gpu-wave.json`.

## Remaining acceptance

The complete acquisition window remains unmeasured at this commit.
Spatial/thickness/time convergence,
Lamb-wave dispersion, representative defects/frequencies, coupled or calibrated
PZT voltage and physical agreement remain required. No research bank is
accepted and no full training ran. `ready_for_training_data` remains false.
