# Preliminary PZT timing comparison

Compared saved simulation voltage-reference traces with the actual top and
bottom PZT recordings stored alongside them. No new propagation runs were needed.
See `examples/compare_pzt_recordings.py` and the full numerical receipt in
`docs/evidence/pzt-measured-timing-comparison-20261117.json`.

Percent difference = 100 * abs(simulated flight time - measured flight time)
/ measured flight time. These are packet-envelope timing proxies, not overall
accuracy, amplitude agreement or validated identification of pure wave modes.

| Frequency | Fast S0-window timing difference, top/bottom | Slow A0-window timing difference |
|---|---|---|
| 50 kHz | 1.3–1.7% | 2.8–6.8% |
| 100 kHz | 4.1% both | 0.7–5.3% |
| 150 kHz | 1.3% both | Unresolved: simulated peaks hit window boundary |
| 200 kHz | 5.3–5.7% | Top 1.1%; bottom unresolved |
| 250 kHz | 5.7% both | Top 1.7%; bottom unresolved |

For unresolved bottom channels at 200/250 kHz, measured maxima occur at the
window edge. At 150 kHz, reporting the bottom channel's nominal 0% difference
would be misleading because both maxima occur at the imposed boundary.
No window was adjusted after seeing results to obtain a better agreement.

Method: common 5 MHz grid over 350 us, baseline from first 20 us, Hilbert
envelope padded to 8192 samples, fixed carrier-theory travel-time window
through travel time plus five cycles. Half the excitation duration is
subtracted from the envelope-peak time. Channels are assessed independently
because wiring/poling signs are unknown. Slow-wave windows may contain
reflected fast waves or overlapping packets; further waveform/mode analysis
is needed before treating their maxima as direct A0 arrivals. Ideal
electrical loading and omitted transducer/glue mechanics remain limitations.

The fast packet timing differs by approximately 1.3–5.7%. The slow packet
comparison is mixed and partly unresolved. A single overall accuracy
percentage is not supported, and these results do not establish training
data readiness or prove that the earlier FBG mismatches are fixed.
