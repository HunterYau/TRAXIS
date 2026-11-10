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
The frozen profile study completes all 2,592 cases in 31.2966 monitored worker
seconds, with 256.91 MiB peak combined RSS, 48.0% system RAM and at least
8,439.11 MiB available. It stays inside its 512 MiB/180-second caps and the
usual system guards. No training is involved.

## Measured profile-band result

The 5-250 kHz band retains 99.99511498% of the nominal force spectral energy,
above its predeclared 99.9% requirement. The source-time energy is 3.75e-6
N^2 s. This does not quantify received-wave or electrical energy.

All 234 outside-first-zone cases are assessed by analytic profile support;
219 have a nonlowest dominant eigenmode. In 150 cases more than one mode is
needed, with up to three selected. Minimum captured profile mass is
99.0402483%. No nearest-frequency matching is used. All 96 prospective finest
cases pass; maximum support frequency error is 0.0300180%.

For standard Q4 GLL with one thickness cell, maximum support frequency errors
over the twelve frequencies and both modes are:

| Uniform thickness | Lateral spacing | Worst frequency error | 0.05% design screen |
| --- | ---: | ---: | --- |
| 3 mm | 5 mm | 0.0780050% | Fail |
| 3 mm | 3.333 mm | 0.00730436% | Pass |
| 3 mm | 2.5 mm | 0.00378979% | Pass |
| 1 mm | 5 mm | 2.42064% | Fail |
| 1 mm | 3.333 mm | 0.222969% | Fail |
| 1 mm | 2.5 mm | 0.0302558% | Pass |

Thus the earlier 5 mm carrier result cannot be used as a band-wide mesh
acceptance. The 3.333 mm candidate is worth a full-size intact-plate cost
measurement; locally thinned/recess cases need their own finer resolution or
another verified choice. These are sampled uniform-waveguide design screens,
not finite-plate wave errors, recess scattering or physical acceptance.

Every selected eigenpair and profile overlap was audited from the raw archive,
reconstructing the small operators and analytic target. Maximum selected
eigenpair relative residual is 5.55e-8. Maximum projection error in the study
is 2.35e-15, analytic Navier residual 2.90e-11 and traction residual 8.44e-14.
The audit verifies complete case counts, exact support selection, retained
mass and every unchanged frequency criterion.

`outputs/tensor-profile-band-20261109` retains the source-bound plan, snapshots,
monitor and raw proofs. The collector is
`outputs/tensor-profile-tools-20261109/collect_profile.py`.
`evidence/lamb-profile-debug.json` binds those records, all 54 GLL summary
groups, the twelve-test group and retained derivative-fixture failure. Its
fingerprint is `8691f435abebf327bb857fc5a39af988372c0853c604dc557a2ad7273605c990`.

The next decision uses actual profile-support errors and retained source
energy, then requires an accelerated action, full-size capacity/cost and
finite-plate validation. It accepts no data and makes no PZT or physical claim.
