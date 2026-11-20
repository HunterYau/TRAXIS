# Restored ideal experiment settings

The selected next run is `outputs/simulation-runs/run-010-wandowski-restored-ideal-properties/`.
Its configuration is `configs/pzt/resolved-wandowski-ideal-restored.json`, copied
from the frozen run 007 ideal model, with all five frequencies (50-250 kHz)
scheduled from the beginning. No simulation was started when selecting it.

Restored physical settings:

- Perfect PZT-to-plate ties, without the assumed flexible adhesive interface/mass.
- Ideal open-circuit receivers, without the assumed scope/cable capacitance or resistance.
- One-volt peak source shaped by the recorded generator waveform, without the
  newer assumed X10 amplifier gain or delay.
- Zero damping, matching the previous ideal runs 006/007.

The plate geometry/material, three resolved PZT bodies and their mechanical
backreaction, mesh, 350-us duration and recorded experiment data match run 007.
This is still an exploratory model, not a validated reconstruction of real
bond/instrument properties. Better timing alone would not establish accurate
amplitude or justify training-data acceptance.

The newer waveform comparisons, collective text results, progress display,
resource policy, output organization and interrupted-run handling remain active.
Arrival timing now uses [adaptive packet windows with fixed-window diagnostics](adaptive-arrival-windows.md).
The unstarted prepared receipt and source snapshot were refreshed; the earlier
preparation is preserved in `provenance/prepared-before-adaptive-arrivals.json`.
The realistic-settings configuration and its completed results are retained in
`configs/pzt/resolved-wandowski-v5.json` and
`outputs/simulation-runs/run-009-wandowski-realistic-bond-damping/`.

Run `examples/run_resolved_pzt.cmd` when ready to start, or use
`python -m examples.resolved_pzt_queue --run` with the project Python runtime.
The default progress viewer now points at run 010. To view the prior completed
realistic batch, pass its folder explicitly with `--root`.

The current numerical readiness evidence includes historical perfect-bond mesh
checks and current backward-compatible lossless/open-port regressions. Restoring
these settings does not create a new full-plate physical validation claim. New
unit checks confirm the frozen physical settings and actual worker wiring; their
mock integrator does not perform a GPU solve.
