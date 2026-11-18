# PZT comparison using the paper's signal separation

Applied section 5 equations 13/14 of the author manuscript:
https://zenodo.org/records/18937842 . Add top and bottom recordings for the
fast (S0) wave; subtract bottom from top for the slow (A0) wave. The paper
explicitly describes in-phase measured S0 and opposite-phase measured A0,
providing support for this fixed sign convention. Simulation uses its
declared +z observer convention. No fitted channel gain, polarity reversal,
time shift, or waveform change was applied. Channels were not individually
normalized before combining. This applies addition/subtraction, not the
paper's separate mode-indicator algorithm.

Analyzed the same saved five experiments. Entire 350 us signals were plotted;
timing extraction retains the earlier common 5 MHz grid, baseline, Hilbert
processing, fixed theoretical packet windows and half-burst subtraction.
Thus percentages retain the prior flight-time definition. Lower is better.

| Frequency | Fast-wave timing difference | Slow-wave candidate timing difference |
|---|---|---|
| 50 kHz | 1.28% | 4.03% |
| 100 kHz | 4.07% | 4.54% |
| 150 kHz | 1.28% | 4.48%, tentative |
| 200 kHz | 5.34% | 0.00%, tentative |
| 250 kHz | 5.71% | 0.60%, tentative |

All extracted peaks are now inside their search windows. The slow-wave
measured/simulated peak times (us from excitation start) are respectively
233.4/240.8, 166.0/172.4, 141.6/136.0, 118.8/118.8 and 110.6/111.2.
Fast-wave measured/simulated peaks are 96.8/96.2, 69.2/71.0, 63.4/62.8,
61.2/58.6 and 59.0/56.2.

Subtraction strongly clarifies the 50/100 kHz slow wave. Above 140 kHz, the
paper reports the direct A0 mode is not detected clearly with these PZTs.
Residuals from unequal sensors/bonding/wires, noise and imperfect fast-wave
cancellation can remain. The candidate high-frequency percentages are saved
for transparency, but the receipt leaves reportable_timing_difference_percent
null and marks them above_reported_A0_detection_range. A 0.00% candidate
merely means the selected peaks occupy the same 0.2 us sample bin, not a
perfect match of the signals. Neither the paper's observation nor this
label establishes a universal hard frequency cutoff.

Full-record envelope correlations of the difference signals are approximately
0.941, 0.948, 0.607, 0.435 and 0.745 in ascending frequency order. In particular,
the 200 kHz 0.00% peak difference does not imply an accurate entire waveform.
These correlations are diagnostic context, not accuracy percentages.

Verification: source NPZ files matched their completed-result hashes; an
independent manufactured mixture verified that addition cancels an opposite-
sign component and subtraction cancels a same-sign component; the rendered
ten-panel plot was visually inspected. No plate simulations were rerun and
no frozen simulator source/configuration was changed.

Analysis: examples/compare_pzt_paired.py. Receipt and figure are under
docs/evidence/pzt-paired-comparison-20261117.{json,png}. Existing comparison
receipts remain intact. This improves the analysis but does not provide a
controlled old/new simulator comparison or establish training readiness.
