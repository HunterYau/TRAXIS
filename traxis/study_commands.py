"""Explicit planning, bounded advances and frozen architecture comparisons."""

import json
from pathlib import Path


def add_study_commands(sub):
    plan = sub.add_parser("plan-study", help="plan matched architecture jobs and a train-only signal scale; no optimization")
    plan.add_argument("--dataset", type=Path, required=True)
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    advance = sub.add_parser("advance-study", help="advance selected study jobs within an explicit update budget")
    advance.add_argument("--plan", type=Path, required=True)
    advance.add_argument("--jobs", nargs="+")
    advance.add_argument("--max-new-updates", type=int, default=1)
    advance.add_argument("--baseline-fits", type=int, default=0)
    freeze = sub.add_parser("freeze-study", help="check matched exposure and freeze completed individual/ensemble comparisons")
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--split", choices=("validation", "test"), default="validation")
    freeze.add_argument("--budgets", type=int, nargs="+", required=True)
    freeze.add_argument("--random-seeds", type=int, nargs="*", default=[17, 29, 43])
    freeze.add_argument("--initial-actions", type=int, nargs="*", default=[])
    freeze.add_argument("--threshold-mm", type=float, default=.1)
    freeze.add_argument("--output", type=Path, required=True)
    for command in (plan, advance, freeze):
        command.add_argument("--max-seconds", type=float, default=60)
        command.add_argument("--max-rss-mib", type=float, default=1024)


def run_study_command(args, guard):
    from traxis.studies import advance_study, create_study, freeze_study_comparisons
    guard.check(force=True)
    if args.command == "plan-study":
        if args.config.stat().st_size > 1024**2:
            raise MemoryError("study configuration exceeds 1 MiB")
        specification = json.loads(args.config.read_text(encoding="utf-8-sig"))
        result = create_study(args.dataset, specification, args.output, guard)
        return {"plan": str(args.output / "study.json"), "jobs": len(result["jobs"]),
                "controlled_pairs": len(result["controlled_pairs"]),
                "planned_neural_updates": sum(j["training"]["max_steps"] for j in result["jobs"] if j["kind"] == "neural"),
                "updates_run": 0, "fingerprint": result["fingerprint"]}
    if args.command == "advance-study":
        result = advance_study(args.plan, job_ids=tuple(args.jobs) if args.jobs else None,
                               max_new_updates=args.max_new_updates, baseline_fits=args.baseline_fits, guard=guard)
        return {key: result[key] for key in ("passed", "new_updates", "baseline_fits", "jobs")}
    result = freeze_study_comparisons(args.plan, args.output, split=args.split, budgets=tuple(args.budgets),
                                      random_seeds=tuple(args.random_seeds), initial_actions=tuple(args.initial_actions),
                                      threshold_mm=args.threshold_mm, guard=guard)
    return {"output": str(args.output), "individual_predictors": len(result["predictors"]),
            "ensemble_plans": len(result["ensembles"]), "replay_has_run": False}
