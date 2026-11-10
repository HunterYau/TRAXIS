# Analytic Lamb profiles and folded spectral support

The tensor design at `bb52be2` withheld 72 target phases outside the first
cell Brillouin zone. A lowest eigenvalue cannot identify the intended physical
wave there. The new profile protocol compares the finite-element spectrum
with an independent analytic displacement field, without selecting whichever
eigenfrequency is nearest to the target.

The field construction follows the scalar/SV-potential treatment of a
traction-free isotropic plate, consistent with the
[analytical wave-structure comparison](https://www.mdpi.com/2076-3417/13/3/1706).
Our implementation derives its coefficients from the traction equations and
checks the resulting fields directly against the elastic PDE. The current
nominal isotropic aluminum constants remain assumptions, not measured data.

## Analytic field checks

`traxis/lamb_profiles.py` uses `exp(i k x - i omega t)` and centered thickness
coordinate `z`. With scalar potential `phi` and SV potential `psi`,

```text
ux = i k phi - psi'
uz = phi' + i k psi
```

S0 uses even `phi` and odd `psi`; A0 uses odd `phi` and even `psi`. Their
vertical wavenumbers satisfy the bulk longitudinal/shear equations. A complex
two-by-two SVD nullspace enforces both surface traction conditions at the
independently computed Rayleigh-Lamb wavenumber. Nondimensional coordinates
reduce coefficient scaling problems. The displacement's arbitrary amplitude
is normalized for comparison, without fitting a numerical waveform.

Analytic first/second derivatives are returned explicitly. Tests check the
coupled Navier equations, both free surfaces, physical parity and independent
finite-difference derivatives for the declared thickness/frequency range.
The first two-point second-derivative fixture used a very small step and
failed at 1.97e-6 relative because of cancellation. A fourth-order stencil
with a larger step passes the unchanged 1e-6 screen. That initial log and
test source are retained; the analytic field was unchanged.

## Frozen support and source-band protocol

`configs/tensor-profile-study.json` declares Q2/Q3/Q4, 5/3.333/2.5 mm cells,
1/2/4 thickness cells, 3/1 mm stock, both modes and both stiffness rules.
Twelve frequency samples span 5 to 250 kHz. All 2,592 cases must exist;
the prospective finest set contains 96 cases. Empty lists, relaxed coverage
and changed discrepancy criteria are rejected.

For each case, the analytic traveling profile is sampled at the tensor nodes
and reduced into the verified periodic/parity space. Projection error must be
below 1e-10. The complete generalized eigensystem accounts for its mass norm.
Modes are selected in descending profile mass overlap until at least 99% is
represented. Eigenfrequencies are not inputs to that selection. The design
error is the largest relative frequency error among the selected modes,
against the unchanged 0.05% frequency screen. A split profile cannot pass
merely because one nearby or high-overlap eigenmode looks accurate.

The finite two-cycle, 100 kHz Hann force has an analytic Fourier transform.
It is tested against the existing `tone_burst` by direct integration and
against Parseval energy. The declared band must retain at least 99.9% of the
nominal force spectral energy. This is force energy, not received-wave energy
or PZT voltage. Discrete frequency samples are a design study; full-window
waveform convergence remains required.

Twelve selected profile/tensor tests pass without skips, including physical
fields, periodic parity, force spectrum, support selection and protocol
rejection. They are separate from the historical complete 268-test suite.
The measured profile study is pending at this protocol checkpoint. It will
run in a source/runtime-bound CPU worker with 512 MiB and 180-second caps,
system RAM below 85% and at least 2 GiB available. No training is involved.

The next decision uses actual profile-support errors and retained source
energy, then requires an accelerated action, full-size capacity/cost and
finite-plate validation. It accepts no data and makes no PZT or physical claim.
