"""Freeze, fit and assess post-hoc prediction intervals without training."""

from pathlib import Path


def add_uncertainty_commands(sub):
    freeze = sub.add_parser("freeze-calibration", help="freeze interval settings before calibration replay")
    freeze.add_argument("--evaluation-plan", type=Path, required=True)
    freeze.add_argument("--alpha", type=float, default=.1)
    freeze.add_argument("--std-floor-mm", type=float, default=.05)
    freeze.add_argument("--mode", choices=("debug", "research"), default="debug")
    freeze.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser("fit-calibration", help="fit a parent-level conformal order statistic on reserved plates")
    fit.add_argument("--plan", type=Path, required=True)
    fit.add_argument("--evaluation", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    assess = sub.add_parser("assess-calibration", help="measure frozen interval coverage on held-out replay")
    assess.add_argument("--calibration", type=Path, required=True)
    assess.add_argument("--evaluation", type=Path, required=True)
    assess.add_argument("--output", type=Path, required=True)
    for command in (freeze, fit, assess):
        command.add_argument("--max-seconds", type=float, default=60)
        command.add_argument("--max-rss-mib", type=float, default=512)


def run_uncertainty_command(args, guard):
    from traxis.uncertainty import evaluate_calibration, fit_calibration, freeze_calibration_plan
    guard.check(force=True)
    if args.command == "freeze-calibration":
        result = freeze_calibration_plan(args.evaluation_plan, args.output, alpha=args.alpha,
                                         std_floor_mm=args.std_floor_mm, mode=args.mode)
        return {"output": str(args.output), "fingerprint": result["fingerprint"], "mode": args.mode}
    if args.command == "fit-calibration":
        result = fit_calibration(args.plan, args.evaluation, args.output, guard)
        return {"output": str(args.output), "passed": result["passed"], "debug_only": result["debug_only"],
                "independent_parents": result["independent_parents"], "finite_quantile": result["finite_quantile"],
                "quantile": result["quantile"]}
    result = evaluate_calibration(args.calibration, args.evaluation, args.output, guard)
    return {"output": str(args.output / "coverage.json"), "passed": result["passed"],
            "debug_only": result["debug_only"], "independent_parents": result["independent_parents"],
            "empirical_parent_simultaneous_coverage": result["empirical_parent_simultaneous_coverage"]}
