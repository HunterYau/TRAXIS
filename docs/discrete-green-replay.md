# Discrete Green replay for mechanical wave traces

One fixed linear mechanical system can reuse its impulse response across
several force waveforms. The new helper performs this operation on the
original fine time grid using float64 CPU FFTs. Direct small-plate solves and
a retained full-size Q3 control verify the force indexing and output traces.
This is an input-reuse check; research-bank and physical acceptance remain
separate requirements.

## Discrete input convention

For fixed mass M, stiffness K, unit-total-force load b and zero initial state,
the displacement recurrence of the existing velocity Verlet method is

`u[n+1] = (2I - dt^2 M^-1 K) u[n] - u[n-1] + dt^2 M^-1 b f[n]`.

The supported force starts at `f[0] = 0`, so `u[0] = u[1] = 0`. Record g
using a one-time-step force of 1 N at sample 1, with all other samples zero.
For a linear displacement probe, the requested trace is

`y[n] = sum(g[n-j+1] * f[j], j=1..n-1)`.

The implementation convolves g with `force[1:]`, zero pads to at least
`2 * sample_count - 2`, and retains the original sample count. Its first
two output samples are the exact zero initial conditions. The recorded g
already includes the discrete integrator's force/time-step scaling; its
units are meters per Newton of this one-step input.

The helper rejects nonzero initial samples, mismatched grids/lengths,
complex/nonfinite inputs and an excessive FFT workspace before starting
the transform. The default additional-workspace screen is 512 MiB.
[SciPy's real FFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfft.html),
[inverse real FFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.irfft.html)
and [padding-size helper](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.next_fast_len.html)
provide the transforms.

This API returns displacement traces. It assumes identical material,
geometry, boundaries, mesh, time step, source covector and receive covectors
across the reused inputs. A placement/contact-radius or physical-operator
change requires another response. Force frequency and amplitude can vary
on the same recorded grid. Keep the fine samples through convolution;
select any declared output stride afterward. Identified PZT/electrical
frequency response remains a separate calibration requirement.

## Checks completed on November 15

Seven tests pass. An independent coupled matrix central-difference recurrence
checks general force sequences and signed probes. Other tests exercise the
first/last force samples, zero-state convention, workspace rejection and
the binding between a retained forward run and its passing CPU receipt.
These metadata fixtures do not serve as forward-physics evidence.

The actual CUDA plate driver runs the 80 by 60 by 3.175 mm Q4 fixture through
all 8,064 steps / 31.5 microseconds. Synthesized 80/100/150 kHz two-cycle
traces match three fresh direct solves within 4.34e-15 relative L2. The
Green-state CPU force agrees within 5.38e-15, mass exactly, and modified
energy varies by 2.46e-14. A separate retained-window path also passes
against the earlier CPU-checked small control.

The full-size 500 by 500 by 3.175 mm intact Q3 check advances all 196,608
steps / 500 microseconds. It reuses the archived physical load/probe arrays
and binds the retained direct trace, original hashes and passing CPU check.
The three 100 kHz trace discrepancies are 1.93e-14, 1.30e-14 and 2.62e-14
relative, below the predeclared 1e-8 replay limit. Independent CPU force,
mass, probe and energy checks also pass. The impulse advance takes 889.119
seconds; FFT synthesis takes 0.0424 seconds. The overall verification takes
980.038 seconds and peaks at 663.32 MiB RSS / 71.0% system RAM. Concurrent
GPU work affects those wall times; they are not a standalone speedup claim.

The impulse remains a discrete numerical input. Its high-frequency content
is not a validation of high-frequency continuum propagation. The successful
full-size replay is at Q3; its separate spatial accuracy is governed by the
intact refinement study. Full-bank generation, other configurations and
PZT/physical agreement remain open. All receipts retain
`ready_for_training_data=false`.

## Commands

```powershell
.\.venv\portable-runtime\python.exe -m unittest discover -s tests -p test_discrete_green.py -v
.\.venv\portable-runtime\python.exe -m examples.verify_green_synthesis --output outputs/green-small-check
.\.venv\portable-runtime\python.exe -m examples.verify_green_synthesis --reference-run outputs/embedded-wave-intact-q3-n196608-20261114-v1 --cpu-check outputs/embedded-wave-intact-q3-n196608-cpu-20261114-v1 --output outputs/green-full-check
```

Use fresh output directories. Plans seal controls, source snapshots, package
identities, force conventions and numerical tolerances before execution.
Raw input forces, impulse/direct/replayed traces and CPU/GPU final forces
are retained. Source/runtime and retained-input hashes are checked again
after the computation. The compact receipt is
`docs/evidence/discrete-green-replay-20261115.json`.

## Force-scale wording

`force_scale_n=1` multiplies the unnormalized Hann-windowed sine from
`tone_burst`. For the two-cycle controls its continuous peak magnitude is
about 0.869619 N. It is a 1 N scale parameter, rather than a post-window
peak of exactly 1 N. Stored cases and force samples keep their original
values; the convergence/replay results use those unchanged inputs.
