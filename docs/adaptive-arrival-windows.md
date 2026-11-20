# Adaptive arrival windows

The old window already depended on frequency and predicted mode group velocity.
It could not follow the observed packet boundaries or move past its computed edges.
The new method independently locates each measured/simulated S0/A0 packet without
fitting shifts, amplitudes or time scaling to improve agreement. Arrival remains
the envelope peak minus half the source burst duration, not first-motion onset.

## Research and declared project choices

[Zeng et al., Sensors 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5087411/)
use a Hilbert envelope and a 0.1 threshold to define guided-wave packet duration.
We adapt that boundary idea, not their full amplitude-dispersion compensation method.
[Li et al., Materials 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5554029/)
use group velocity to gate direct packets and reduce reflection mixing.

Our additional constants are engineering choices, not calibrated uncertainty bounds:

- Eleven group-speed predictions across the approximate Hann-burst main lobe,
  f*(1 +/- 2/cycles). Configuration supplies material, mode, thickness and path.
- Search from 75% of earliest theoretical flight to 125% of latest flight plus
  source duration; bounded by saved-record end and an estimated earliest same-mode
  single-edge reflection. Rectangle image-source paths supply reflection distance.
- Threshold: maximum of 10% of envelope peak and baseline median plus six scaled
  median absolute deviations (MAD scale 1.4826); baseline uses first 20 us.
- Packet boundaries follow threshold crossings. Comparable secondary peaks
  (70% height and half-burst separation), clipping and low signal are flagged.

## Method selection

Save fixed and adaptive diagnostics. Prefer adaptive when a unique uncut packet
agrees with the old valid peaks or recovers a packet rejected at an old window
edge. Retain the original fixed estimate when adaptive boundary checks reject
exactly the same previously valid peaks, within one 0.2-us sample. Do not fall
back for noise or competing packets. Disagreement remains unresolved. Selection
never uses the lower paired error percentage.

Synthetic tests cover known timing beyond the old gate, a stronger late reflection,
noise rejection, independent timing shifts and competing packets. Saved-record
reanalysis is under `outputs/result-analyses/adaptive-arrival-window-v2/` (v1
preserves the initial adaptive-only diagnostics before adding fixed fallback).
Those records have no independently labelled true arrival timestamps, so they
cannot establish better true timing accuracy. Most peaks remain the same and
some high-frequency realistic cases remain unresolved.

The source's A0 detection-range warning above 140 kHz remains. Imperfect mode
separation and converted/reflected-mode contamination remain possible. An envelope
packet is not proof of pure S0/A0. Original simulation outputs are preserved.
