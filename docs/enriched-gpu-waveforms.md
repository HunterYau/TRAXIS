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

## Remaining acceptance

Full-size GPU construction, explicit CFL/cost and the complete acquisition
window remain unmeasured at this commit. Spatial/thickness/time convergence,
Lamb-wave dispersion, representative defects/frequencies, coupled or calibrated
PZT voltage and physical agreement remain required. No research bank is
accepted and no full training ran. `ready_for_training_data` remains false.
