"""Predeclared numerical refinement and physical-unit trace plots."""

import json
from pathlib import Path


def add_waveform_commands(sub):
    sweep = sub.add_parser("check-waveforms", help="measure mesh/time trace changes; failed screens return exit code 2")
    sweep.add_argument("--plan", type=Path, required=True)
    sweep.add_argument("--index", type=int, default=0)
    sweep.add_argument("--settings", type=Path, required=True)
    sweep.add_argument("--output", type=Path, required=True)
    sweep.add_argument("--cache-dir", type=Path, help="reuse completed, integrity-checked solver groups across bounded attempts")
    plot = sub.add_parser("plot-waveforms", help="plot checked float64 refinement probes without new solves")
    plot.add_argument("--sweep", type=Path, required=True)
    plot.add_argument("--records", type=int, nargs="+", default=[0, 12])
    plot.add_argument("--output", type=Path, required=True)
    for command in (sweep, plot):
        command.add_argument("--max-seconds", type=float, default=60)
        command.add_argument("--max-rss-mib", type=float, default=512)


def run_waveform_command(args, guard):
    from traxis.waveform_checks import WaveformSweepSettings, plot_waveform_sweep, run_waveform_sweep
    guard.check(force=True)
    if args.command == "plot-waveforms":
        result = plot_waveform_sweep(args.sweep, args.output, tuple(args.records), guard)
        return {"completed": True, "figures": result["files"], "ready_for_training_data": False}
    from traxis.scenarios import load_scenario_plan
    cases = load_scenario_plan(args.plan)
    if not 0 <= args.index < len(cases):
        raise ValueError("scenario index is outside the plan")
    if args.settings.stat().st_size > 1024**2:
        raise MemoryError("refinement settings exceed 1 MiB")
    settings = WaveformSweepSettings.from_dict(json.loads(args.settings.read_text(encoding="utf-8-sig")))
    result = run_waveform_sweep(cases[args.index], settings, args.output, guard, cache_dir=args.cache_dir)
    return {"completed": result["completed"], "discretization_screen_passed": result["discretization_screen_passed"],
            "spatial_screen_passed": result["spatial_screen_passed"], "temporal_screen_passed": result["temporal_screen_passed"],
            "observed_temporal_order": result["observed_temporal_order"], "wall_seconds": result["wall_seconds"],
            "ready_for_training_data": False, "report": str(args.output / "waveform-report.json")}
