# Five measured plate-wave timing comparisons

Status: 2 of 5 complete. The remaining three are queued serially.

Google Doc entries go to **Science Log Info**, the user's notes document
for preparing the formal log later:
https://docs.google.com/document/d/1sER_SaOLuuthx7hU7C40QxHHgPvBV71T_ILr1cGGZl8.
The actual science log has not been edited. The user clarified this
distinction during test 2; the connected document title was verified.

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
| 2 | 100 kHz | 145.2 us | 131.4 us | -13.8 us | -9.5041% | 43.16 min |
| 3 | 150 kHz | Pending | Pending | Pending | Pending | Running |
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

## Test 2: 100 kHz

Both packet peaks lie inside the fixed window (139.566 to 189.566 us).
Measured and simulated peaks are 170.2 and 156.4 us. Subtracting the 25 us
half-pulse gives flight times 145.2 and 131.4 us. Packet group speeds from
this extraction are 1721.76 and 1902.59 m/s. The model arrived 13.8 us
early, a -9.5041% flight-time difference. This is a significant discrepancy
for this observable; the cause has not been isolated.

The visually checked plot shows two strong lobes in the predicted packet,
while the measured main packet is more compact. The fixed envelope-maximum
rule selected the earlier predicted lobe. Do not shift the window, select
the later lobe or tune parameters to improve the reported agreement.
This packet-shape difference limits interpreting its peak as a pure
material group-speed measurement. Full transducer transfer models and full
spatial/time convergence remain open; the small carrier-frequency Bloch
error alone does not establish full-packet accuracy.

All 65,536 steps completed. Independent CPU final-force error is
3.2664e-13 relative, and post-burst modified-energy variation is
3.9486e-16 relative. Native wall time is 2589.772 s. These numerical checks
passed despite the experimental discrepancy. Observed worker/monitor
peaks: system RAM 64.4%, owned process tree 2009.31 MiB, GPU memory
3520 MiB and GPU temperature 76 C; limits were respected.

Evidence: [receipt](evidence/experimental-trial-2-100khz-20261116.json),
[portable waveforms](evidence/experimental-trial-2-100khz-20261116.npz),
[comparison plot](evidence/experimental-trial-2-100khz-20261116.png).
No scientific parameters were changed after this result. Test 3 uses the
same frozen protocol at 150 kHz.
