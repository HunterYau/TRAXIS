# Prepared PZT-to-PZT validation queue

Status: **prepared, not run**. This is a new experiment path; the completed FBG
comparisons and their frozen receipts remain historical evidence.

The selected reference is Wandowski et al., *Elastic wave mode identification
and separation based on piezoelectric and fibre Bragg grating sensors*, Smart
Materials and Structures 34 (2025), 105015,
[published article](https://doi.org/10.1088/1361-665X/ae0d1d),
[author manuscript](https://zenodo.org/records/18937842), and
[authors' measured data and README](https://zenodo.org/records/18955122).
The locally downloaded MATLAB data are checksum verified before preparation and
each worker. Numerical SEM traces in the archive are **not** treated as experiment.

This reference was chosen because it supplies real electrical PZT recordings,
the actual generator waveform, sensor placement, named PZT material and disc
dimensions, and a published description. It is a strong reference, but it does
not identify every parameter necessary for an exact digital reproduction.

## Source-to-model map

| Property | Reference and treatment |
|---|---|
| Plate | 1000 x 1000 x 1 mm aluminum, specimen 1 |
| Alloy | Not specified; do not call it 6061-T6 |
| Host elastic properties | Paper's nominal model values: E=71 GPa, nu=0.33, density=2700 kg/m3; not independently measured specimen properties |
| Actuator | Top center (0.5,0.5) m; NCE51 disc, diameter 10 mm, thickness 0.5 mm |
| Receivers | Top and bottom pair, 250 mm to the right: (0.75,0.5) m |
| PZT constitutive properties | Existing independently sourced Noliac Ver1404 NCE51 tensor; nominal material, not the measured properties of these specific discs |
| Bond | Cyanoacrylate reported; thickness and mechanical properties unknown. Ideal perfect transfer approximation; no explicit bond domain |
| Excitation | Recorded generator trace, five Hann cycles; normalized to unit peak reference voltage. Actual amplified actuator voltage is not calibrated |
| Recording | 5 MHz, 50 averages; MATLAB columns 3 and 4 (zero-based 2 and 3) are PZT top/bottom. Columns 1 and 2 are FBG and excluded |
| Equipment | NI PXIe-5413, Krohn-Hite 7500, NI PXIe-5105. Actual input/load/cable transfer remains unknown |
| Boundary | Free plate, zero damping model; actual support details not established |
| Variability | None: deterministic reconstruction, not training domain randomization |

The manuscript's setup diagram and measurement section give the geometry; the
dataset README identifies channel order. The new configuration is
`configs/pzt/wandowski-2025-plate1.json`.

## What is connected

The new driver uses **PZT area-integrated strain charge**, not the old FBG mean
axial strain. Each circular probe uses both in-plane normal strain components
and the relevant material coupling. Its spatial integration preserves physical
cancellation. The source is the reciprocal planar piezoelectric force vector
under a prescribed unit-peak reference voltage with the recorded waveform's
shape. For the port convention Q=C V-g^T u, its external force is -g V.

This is a quasistatic, perfectly bonded, one-way thin-disc approximation.
It does not add the disc's body inertia/stiffness, electrical backreaction,
radial resonance, or glue mechanics to the large plate operator. The 0.5 mm PZT
is substantial relative to this 1 mm host, so omitted mechanics may matter.
Using a full material tensor to derive the planar coefficients does not turn
this into a full three-dimensional PZT assembly.

Raw output includes `strain_charge_c`, explicitly labelled
`ideal_open_reference_voltage_v` (zero external capacitance, infinite resistance),
and the two actual experimental PZT voltage channels. This mathematical ideal
is not an assumed real instrument impedance. Absolute amplitude comparisons
and top/bottom mode sums are not justified without gain/load/poling information.
Inspect timing and normalized shape with these limits; do not fit away shifts
or claim validated training data. There is no experimental acceptance threshold.

## Frequencies and manual launch

Future progress displays should use terminal text (user preference,
November 17), with no GUI. In a separate terminal from the simulator:

```powershell
.\.venv\portable-runtime\python.exe -m examples.pzt_terminal
```

This read-only display shows a progress bar, frequency/test number, steps,
elapsed time and approximate ETA. It prints saved results after each
experiment: completion status, runtime, CPU/GPU force difference, energy
drift and output location. It explicitly says when comparison with real
measurements is still pending analysis. It can also print existing saved
results; `--once` takes a snapshot and `--root PATH` selects another queue.
Closing it leaves the independent simulation running. No extra package is
needed. Redirected output uses complete lines instead of terminal redraws.

The serial queue is **50, 100, 150, 200, 250 kHz** on the same specimen.
Specimen 1 has no 300 kHz measurement. Specimen 2 has higher-frequency data,
but its extra setup description focuses on replacement FBG sensors. Following
the user's permission to omit 300 kHz, this queue does not mix specimens or
silently extrapolate the first specimen's measured data.

Preparation/preflight uses CPU metadata and data checks only:

```powershell
.\.venv\portable-runtime\python.exe -m examples.wandowski_pzt_queue
```

The local immutable `outputs/wandowski-pzt-plate1/prepared.json` records the
configuration, source/data hashes and five prepared cases. The preparation
command is `--prepare`; it refuses to overwrite an existing receipt.

**Only when simulations are wanted**, launch:

```powershell
.\.venv\portable-runtime\python.exe -m examples.wandowski_pzt_queue --run
```

That command runs one frequency at a time and prints progress, saves raw
waveforms/final state/results after each, and stops on failure. GPU/independent
CPU final-force and post-burst energy checks remain required. No scheduled task
or background job was created. Importing the module or invoking it without
`--run` does not start simulations. Start the queue through this explicit
entrypoint, not the unrelated training launcher.

Create `outputs/wandowski-pzt-plate1/STOP_AFTER_CURRENT` to stop between trials.
Exclusive `queue-owner.json` ownership prevents concurrent or accidental repeat
launch. Completed/partial attempts are never overwritten. A restart needs a
deliberate new attempt directory/version after reviewing saved results.

The queue retains the earlier 250x250x1, order-5 mesh, 65536 steps over 350 us,
with bounded allocations, RAM ceiling 89.9%, GPU temperature limit 80 C and
cooling pauses at 76 C until 72 C. These settings are numerical controls, not
proof of full spatial/time convergence or hardware fidelity. Runtime on this
GPU is not newly measured because no simulation was launched.
