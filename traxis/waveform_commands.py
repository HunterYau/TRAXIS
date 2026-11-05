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
    reference = sub.add_parser('check-reference-waveforms', help='one P2 group with independent mesh/time axes; failed screens return 2')
    reference.add_argument('--plan', type=Path, required=True)
    reference.add_argument('--index', type=int, default=0)
    reference.add_argument('--settings', type=Path, required=True)
    reference.add_argument('--elmer-home', type=Path, required=True)
    reference.add_argument('--transmitter', type=int, default=0)
    reference.add_argument('--frequency-khz', type=float)
    reference.add_argument('--state', choices=['intact', 'current'], default='intact')
    reference.add_argument('--cache-dir', type=Path)
    reference.add_argument('--output', type=Path, required=True)
    plot = sub.add_parser("plot-waveforms", help="plot checked float64 refinement probes without new solves")
    plot.add_argument("--sweep", type=Path, required=True)
    plot.add_argument("--records", type=int, nargs="+", default=[0, 12])
    plot.add_argument("--output", type=Path, required=True)
    reference_plot = sub.add_parser('plot-reference-waveforms', help='plot checked P2 refinement traces without new solves')
    reference_plot.add_argument('--sweep', type=Path, required=True)
    reference_plot.add_argument('--records', type=int, nargs='+', default=[0, 3])
    reference_plot.add_argument('--output', type=Path, required=True)
    for command in (sweep, plot, reference, reference_plot):
        command.add_argument("--max-seconds", type=float, default=60)
        command.add_argument("--max-rss-mib", type=float, default=512)


def run_waveform_command(args, guard):
    from traxis.waveform_checks import WaveformSweepSettings, plot_waveform_sweep, run_waveform_sweep
    guard.check(force=True)
    if args.command in ('plot-waveforms', 'plot-reference-waveforms'):
        result = plot_waveform_sweep(args.sweep, args.output, tuple(args.records), guard,
                                      reference=args.command == 'plot-reference-waveforms')
        return {"completed": True, "figures": result["files"], "ready_for_training_data": False}
    from traxis.scenarios import load_scenario_plan
    cases = load_scenario_plan(args.plan)
    if not 0 <= args.index < len(cases):
        raise ValueError("scenario index is outside the plan")
    if args.settings.stat().st_size > 1024**2:
        raise MemoryError("refinement settings exceed 1 MiB")
    values = json.loads(args.settings.read_text(encoding='utf-8-sig'))
    if args.command == 'check-reference-waveforms':
        from traxis.elmer_plate_refinement import ElmerPlateRefinementSettings, run_elmer_plate_refinement
        settings = ElmerPlateRefinementSettings.from_dict(values)
        result = run_elmer_plate_refinement(cases[args.index], settings, args.elmer_home, args.output, guard,
            transmitter=args.transmitter, frequency_hz=args.frequency_khz * 1000 if args.frequency_khz is not None else None,
            state=args.state, cache_dir=args.cache_dir)
        filename = 'refinement-report.json'
    else:
        settings = WaveformSweepSettings.from_dict(values)
        result = run_waveform_sweep(cases[args.index], settings, args.output, guard, cache_dir=args.cache_dir)
        filename = 'waveform-report.json'
    return {"completed": result["completed"], "discretization_screen_passed": result["discretization_screen_passed"],
            "spatial_screen_passed": result["spatial_screen_passed"], "temporal_screen_passed": result["temporal_screen_passed"],
            "observed_temporal_order": result["observed_temporal_order"], "wall_seconds": result["wall_seconds"],
            "ready_for_training_data": False, "report": str(args.output / filename)}
