# Aluminum damping estimate used for v8

Source: Kanji Ono, Experimental Determination of Lamb-Wave Attenuation
Coefficients (2022), https://doi.org/10.3390/app12136735, Table 1, row G3b.
The measured bulk attenuation slopes for a 25.3 mm 6061-T6 specimen are
4.9 dB/m/MHz longitudinal and 10.3 dB/m/MHz transverse; wave speeds are
6400 and 3170 m/s. These are not measurements of the Wandowski specimen.
That experiment does not identify its aluminum alloy.

For a weakly lossy traveling harmonic wave, amplitude attenuation a in Np/m
and phase speed c give loss factor eta approximately 2*a*c/omega.
At 1 MHz, a = slope*ln(10)/20. Average the longitudinal and transverse
estimates: eta = 0.0011728985 (about 0.1173%). This derived average is an
engineering assumption, not a universal aluminum damping constant.

Use C=alpha*M with alpha=eta*2*pi*f_center. This matches the chosen loss
factor at each burst center; away from that frequency its effective loss
factor is alpha/omega. It is not a broadband constant-loss material and
cannot reproduce separate longitudinal/transverse loss tensors or exact
mode-dependent Lamb attenuation. It uses effective damping on the entire
coupled mass, including the PZT bodies, rather than assigning measured
material loss separately to aluminum and ceramics. It excludes a separate
glue/support/instrument loss model. Settings are configurable in
configs/pzt/aluminum-damping-ono2022.json and recorded in every result.

The velocity damping substep is integrated exactly as
v <- exp(-alpha*dt/2)*v before and after the conservative Verlet step.
This symmetric split is second order and works with consistent coupled
mass. No finished waveform is multiplied by an arbitrary decay curve.
Setting the solver coefficient to zero preserves the original path.

After the applied pulse ends, the modified energy must decrease; it must
no longer pass a zero-energy-drift test. The driver checks positive energy
growth, records the range separately and continues checking final force
against independent CPU assembly. This is not a forced-work identity.

Verification: docs/evidence/damped-verlet-cuda-check.json checks real CUDA
solutions against an exact damped oscillator for diagonal and coupled
consistent mass. Halving dt reduces displacement error by about four;
zero-damping regression conserves energy, and damped unforced energy does
not increase. docs/evidence/damped-pzt-full-size-check.json records a
128-step full-geometry execution check (30.4 s). It covers only 0.684 us,
not the full 350 us or experimental agreement. Old evidence is retained;
the new readiness receipt states this limited scope. No full batch was run.

Reproduce the numerical check in PowerShell from the project folder:

```powershell
$env:CUPY_CACHE_DIR = Join-Path $PWD 'outputs/damped-verlet-check/cupy-cache'
.\.venv\portable-runtime\python.exe -m examples.check_damped_verlet
```

The check writes a new evidence file and refuses to overwrite an existing
one. Existing v6/v7 outputs and snapshots remain unchanged.
