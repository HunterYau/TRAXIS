# Five measured plate-wave timing comparisons

Status: 1 of 5 complete. The remaining four are queued serially.

These are five frequencies on one 1000 x 1000 x 1 mm aluminum specimen.
Alloy and temper are unspecified in the source. This is an exploratory A0
arrival-time comparison using ideal transducer approximations, not full
PZT/FBG voltage calibration, 6061-T6 validation or training-data acceptance.
Protocol and limitations: [experimental setup](experimental-five-20261116.md).
Measured source: https://zenodo.org/records/18955122; author manuscript:
https://zenodo.org/records/18937842. Measured excerpts are CC0 and attributed
to Wandowski, Fiborek, Ahamad, Soman and Malinowski (2025).

| Test | Frequency | Measured flight time | Simulated flight time | Simulated delay | Flight-time difference | Native wall time |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 50 kHz | 190.4 us | 191.6 us | +1.2 us | +0.6303% | 42.98 min |
| 2 | 100 kHz | Pending | Pending | Pending | Pending | Running |
| 3 | 150 kHz | Pending | Pending | Pending | Pending | Queued |
| 4 | 200 kHz | Pending | Pending | Pending | Pending | Queued |
| 5 | 250 kHz | Pending | Pending | Pending | Pending | Queued |

Flight time subtracts half the five-cycle excitation duration from the
packet-envelope peak. Positive delay means the simulation arrived later.
Timing is sampled on the common 0.2 microsecond grid. Measurement geometry,
trigger and transducer uncertainties were not provided, so this table is
not an uncertainty-bounded accuracy estimate.

## Test 1: 50 kHz

Both packet peaks lie inside the fixed extraction window (187.799 to
287.799 us). Measured and simulated peaks are 240.4 and 241.6 us. Equivalent
packet group speeds are 1313.03 and 1304.80 m/s. The CPU recomputation of
final force differs by 1.6468e-12 relative; post-burst modified-energy
variation is 5.5775e-16 relative. All 65,536 planned steps completed.
Native wall time is 2578.812 s. The numerical checks passed; no experimental
pass threshold or full-simulator accuracy percentage is asserted.

Observed peaks across worker and monitor: system RAM 64.6%, owned process
tree 2007.66 MiB, device-wide GPU memory 3562 MiB, and GPU temperature 76 C.
These include observations from separate sampling times. CPU temperature
was unavailable. Resources stayed within the configured guards.

Evidence: [receipt](evidence/experimental-trial-1-50khz-20261116.json),
[portable waveforms](evidence/experimental-trial-1-50khz-20261116.npz),
[comparison plot](evidence/experimental-trial-1-50khz-20261116.png).
The plot was visually checked for layout and signal/window correspondence;
voltage and strain appear in separate panels with their own units.
The FBG observation uses mean axial strain and omits the optical transfer
model; the actuator omits full electromechanical loading. These limitations
can affect packet shape and timing, even when arrival times agree closely.
