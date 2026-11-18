# Full available PZT record comparison

Analyzed all ten top/bottom traces over the full shared sampled interval,
0–349.8 microseconds (350 microsecond simulation), without arrival-selection
windows or fitted time shifts. Baseline is the first 20 microseconds; both
signals use identical Hilbert processing. Source waveform hashes were checked
against completed result files. No simulations were rerun.

| Frequency | Full-record envelope Pearson correlation, top / bottom | Strongest measured peak, top / bottom (us) | Strongest simulated peak, top / bottom (us) |
|---|---|---|---|
| 50 kHz | 0.960 / 0.897 | 235.6 / 228.8 | 240.8 / 241.0 |
| 100 kHz | 0.843 / 0.662 | 165.0 / 161.8 | 71.0 / 71.0 |
| 150 kHz | 0.977 / 0.968 | 63.4 / 63.4 | 62.8 / 62.8 |
| 200 kHz | 0.859 / 0.857 | 61.4 / 61.2 | 58.6 / 58.6 |
| 250 kHz | 0.798 / 0.794 | 59.0 / 59.0 | 56.2 / 56.2 |

Correlation measures similarity of the envelope's rises and falls across the
whole record. It is not a percentage accuracy, amplitude calibration, or
acceptance criterion. Quiet portions and strong packets influence it. Full
voltage correlations are also saved and are lower (approximately 0.235–0.814),
showing that envelope agreement does not guarantee detailed oscillation/phase
agreement. Unknown channel signs are not fitted away.

The full search detects real local maxima rather than clipping a rising signal
at the old window boundary. At 150 kHz, strong later peaks occur at 155.4/155.0
us measured versus 155.0/154.6 us simulated. However, the expected first
right-edge S0 reflection travels 0.75 m, three times the direct path. Its
expected envelope time is 3*(0.25/cg_S0)+2.5/f, approximately 154.9 us at
150 kHz, 150.8 us at 200 kHz and 148.4 us at 250 kHz. The corresponding
observed/simulated peaks are consistent with this reflected fast wave.
They cannot simply replace the unresolved direct slow-wave arrival.

At 200 kHz bottom, the nearest small measured peaks are 103.8 and 132.2 us;
there is no detected local maximum at the simulated 118.8 us peak using the
recorded 2% prominence criterion. At 250 kHz bottom, small measured maxima
occur at 107.8 and 125.0 us around the simulated 111.2 us feature. Thus wider
inspection does not establish a unique matching direct A0 peak in those
channels. At 150 kHz a weak direct-wave feature can overlap other packets;
identification remains uncertain even though the full envelope matches well.

At 100 kHz the strongest measured peak is the later packet near 162–165 us,
while the simulated strongest peak is near 71 us. Both packets exist in both
records, but their relative strengths differ. Subtracting those global peak
times would compare different packets, not establish a 56–57% travel-time
error. The earlier small timing errors did not capture this discrepancy.

Summary: 150 kHz has the closest full-envelope match, 50 kHz also matches
reasonably in envelope, 200/250 kHz have visible timing/shape differences,
and 100 kHz has a substantial relative packet-strength mismatch. The prior
slow-wave ambiguities are now investigated rather than caused solely by an
unexamined cutoff, but cannot all be assigned reliable arrival percentages.
No overall accuracy percentage or training readiness is established.

Reproduction: `python -m examples.compare_pzt_full_record` (refuses to overwrite
the saved JSON receipt). Numerical records, all detected local peaks and
hashes: `docs/evidence/pzt-full-record-comparison-20261117.json`. Figure:
`docs/evidence/pzt-full-record-comparison-20261117.png`. The figure was
visually inspected after rendering.
