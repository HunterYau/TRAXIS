import argparse
from dataclasses import asdict
import json
from pathlib import Path

from traxis.acquisition import action_catalog
from traxis.geometry import estimate_mesh
from traxis.resources import ResourceGuard, ResourceLimits, snapshot


def main() -> None:
    from traxis.experiment_commands import add_experiment_commands
    from traxis.generation_commands import add_generation_commands
    from traxis.uncertainty_commands import add_uncertainty_commands
    from traxis.study_commands import add_study_commands
    parser = argparse.ArgumentParser(description="Bounded guided-wave tomography experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="inspect resources and mesh cost without solving")
    preflight.add_argument("--cells", type=int, nargs=3, default=(250, 250, 6))
    dispersion = sub.add_parser("dispersion", help="independent A0/S0 infinite-plate reference")
    dispersion.add_argument("--frequencies-khz", type=float, nargs="+", default=[50, 75, 100, 125, 150])
    dispersion.add_argument("--thickness-mm", type=float, nargs="+", default=[3.0, 1.0])
    dispersion.add_argument("--output", type=Path)
    verify = sub.add_parser("verify", help="short CPU elastic FEM checks; no training")
    verify.add_argument("--output", type=Path, default=Path("outputs/verification"))
    verify.add_argument("--max-seconds", type=float, default=60)
    verify.add_argument("--max-rss-mib", type=float, default=512)
    importer = sub.add_parser("import-traces", help="import mapped scalar CSV probe exports")
    importer.add_argument("--manifest", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    models = sub.add_parser("verify-models", help="eight CPU single-step debug checks; no training cycle")
    models.add_argument("--output", type=Path, default=Path("outputs/model-verification"))
    models.add_argument("--max-seconds", type=float, default=60)
    models.add_argument("--max-rss-mib", type=float, default=1024)
    workflow = sub.add_parser("verify-workflow", help="short cache-to-replay software pilot; no full training")
    workflow.add_argument("--output", type=Path, default=Path("outputs/workflow-verification"))
    workflow.add_argument("--max-seconds", type=float, default=60)
    workflow.add_argument("--max-rss-mib", type=float, default=1024)
    adaptive = sub.add_parser("verify-policy", help="two-episode adaptive pilot on existing software fixtures")
    adaptive.add_argument("--dataset", type=Path, required=True)
    adaptive.add_argument("--reconstructors", type=Path, nargs="+", required=True)
    adaptive.add_argument("--output", type=Path, default=Path("outputs/policy-verification"))
    adaptive.add_argument("--max-seconds", type=float, default=60)
    adaptive.add_argument("--max-rss-mib", type=float, default=1024)
    calibration = sub.add_parser("verify-calibration", help="bounded reserved-parent interval pilot; no training")
    calibration.add_argument("--dataset", type=Path, required=True)
    calibration.add_argument("--reconstructors", type=Path, nargs="+", required=True)
    calibration.add_argument("--output", type=Path, default=Path("outputs/calibration-verification"))
    calibration.add_argument("--max-seconds", type=float, default=60)
    calibration.add_argument("--max-rss-mib", type=float, default=1024)
    study = sub.add_parser("verify-study", help="one-update matched architecture pilot on existing software fixtures")
    study.add_argument("--dataset", type=Path, required=True)
    study.add_argument("--config", type=Path, default=Path("configs/study-debug.json"))
    study.add_argument("--output", type=Path, default=Path("outputs/study-verification"))
    study.add_argument("--max-seconds", type=float, default=60)
    study.add_argument("--max-rss-mib", type=float, default=1024)
    forward = sub.add_parser("verify-forward", help="bounded CAD-to-elastic-bank checks; no training")
    forward.add_argument("--output", type=Path, default=Path("outputs/forward-verification"))
    forward.add_argument("--max-seconds", type=float, default=60)
    forward.add_argument("--max-rss-mib", type=float, default=512)
    report = sub.add_parser("report", help="render physical-unit figures and tables from completed frozen replay")
    report.add_argument("--evaluations", type=Path, nargs="+", required=True)
    report.add_argument("--labels", nargs="+")
    report.add_argument("--case-count", type=int, default=1)
    report.add_argument("--cases", nargs="+", help="explicit run:episode indices, overriding --case-count")
    report.add_argument("--case-records", type=int, nargs="+")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--max-seconds", type=float, default=60)
    report.add_argument("--max-rss-mib", type=float, default=1024)
    add_experiment_commands(sub)
    add_generation_commands(sub)
    add_uncertainty_commands(sub)
    add_study_commands(sub)
    args = parser.parse_args()
    if args.command in ("plan-study", "advance-study", "freeze-study"):
        from traxis.study_commands import run_study_command
        guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib))
        try:
            print(json.dumps(run_study_command(args, guard), indent=2, allow_nan=False))
        except Exception as exc:
            parser.exit(1, f"Study stopped: {type(exc).__name__}: {exc}\n")
    elif args.command in ("freeze-calibration", "fit-calibration", "assess-calibration"):
        from traxis.uncertainty_commands import run_uncertainty_command
        guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib))
        try:
            print(json.dumps(run_uncertainty_command(args, guard), indent=2, allow_nan=False))
        except Exception as exc:
            parser.exit(1, f"Calibration stopped: {type(exc).__name__}: {exc}\n")
    elif args.command == "report":
        from traxis.reporting import create_report
        guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib))
        try:
            cases = tuple(tuple(int(n) for n in value.split(":")) for value in args.cases) if args.cases else None
            result = create_report(args.evaluations, args.output, labels=args.labels, case_count=args.case_count,
                                   case_records=tuple(args.case_records) if args.case_records is not None else None, cases=cases, guard=guard)
        except Exception as exc:
            parser.exit(1, f"Report stopped: {type(exc).__name__}: {exc}\n")
        print(json.dumps({"passed": result["passed"], "debug_only": result["debug_only"],
                          "figures": len([p for p in result["files"] if p.endswith(".png")]),
                          "index": str(args.output / "index.html")}, indent=2))
    elif args.command in ("plan-scenarios", "export-solver-request", "generate-bank"):
        from traxis.generation_commands import run_generation_command
        guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib))
        try:
            print(json.dumps(run_generation_command(args, guard), indent=2, allow_nan=False))
        except Exception as exc:
            parser.exit(1, f"Generation stopped: {type(exc).__name__}: {exc}\n")
    elif args.command in ("index-data", "train", "train-policy", "calibrate-das", "freeze-evaluation", "evaluate"):
        from traxis.experiment_commands import run_experiment_command
        guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib))
        try:
            print(json.dumps(run_experiment_command(args, guard), indent=2, allow_nan=False))
        except Exception as exc:
            parser.exit(1, f"Experiment stopped: {type(exc).__name__}: {exc}\n")
    elif args.command == "preflight":
        frequencies = (50000., 75000., 100000., 125000., 150000.)
        print(json.dumps({"hardware": snapshot(), "mesh_estimate": asdict(estimate_mesh(tuple(args.cells))),
                          "reciprocal_scan_counts": {str(n): len(action_catalog(n, frequencies)) for n in (8, 12)},
                          "note": "Mesh estimate is an allocation screen, not a converged full-plate mesh recommendation; frequencies are provisional."}, indent=2))
    elif args.command == "dispersion":
        from traxis.dispersion import mode_velocity
        report = {"kind": "infinite_free_isotropic_plate_reference",
                  "note": "Nominal elastic constants; not finite-plate arrival validation or measured PZT response.",
                  "modes": [asdict(mode_velocity(f * 1000, t / 1000, mode))
                            for t in args.thickness_mm for f in args.frequencies_khz for mode in ("A0", "S0")]}
        serialized = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        print(serialized)
    elif args.command == "import-traces":
        from traxis.importers import import_trace_bundle
        guard = ResourceGuard()
        guard.check(force=True)
        bank = import_trace_bundle(args.manifest, guard)
        bank.save(args.output)
        print(json.dumps({"output": str(args.output), "specimen_id": bank.metadata.specimen_id,
                          "records": len(bank.actions), "samples": len(bank.time_s),
                          "validation_status": bank.metadata.validation_status}, indent=2))
    elif args.command == "verify-forward":
        guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib))
        guard.check(force=True)
        try:
            from traxis.forward_verification import verify_forward
            report = verify_forward(args.output, guard)
        except Exception as exc:
            parser.exit(1, f"Forward pilot stopped: {type(exc).__name__}: {exc}\n")
        print(json.dumps({"passed": report["passed"], "ready_for_training_data": False,
                          "wall_seconds": report["wall_seconds"], "report": str(args.output / "forward-report.json")}, indent=2))
        if not report["passed"]:
            parser.exit(1, "Forward pilot failed; inspect its report.\n")
    elif args.command == "verify-study":
        from traxis.study_verification import verify_study
        guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib))
        guard.check(force=True)
        try:
            result = verify_study(args.dataset, args.config, args.output, guard)
        except Exception as exc:
            parser.exit(1, f"Study pilot stopped: {type(exc).__name__}: {exc}\n")
        print(json.dumps({"passed": result["passed"], "new_updates": result["new_updates"],
                          "replay_episodes": result["replay_episodes"], "wall_seconds": result["wall_seconds"],
                          "report": str(args.output / "study-report.json")}, indent=2))
        if not result["passed"]:
            parser.exit(1, "Study pilot failed; inspect its report.\n")
    elif args.command == "verify-calibration":
        from traxis.calibration_verification import verify_calibration
        from traxis.dataset import BankDataset
        guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib))
        guard.check(force=True)
        try:
            result = verify_calibration(BankDataset.load(args.dataset), args.reconstructors, args.output, guard)
        except Exception as exc:
            parser.exit(1, f"Calibration pilot stopped: {type(exc).__name__}: {exc}\n")
        print(json.dumps({"passed": result["passed"], "training_updates": 0, "wall_seconds": result["wall_seconds"],
                          "report": str(args.output / "calibration-report.json")}, indent=2))
        if not result["passed"]:
            parser.exit(1, "Calibration pilot failed; inspect its report.\n")
    elif args.command == "verify-policy":
        guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib))
        guard.check(force=True)
        try:
            from traxis.dataset import BankDataset
            from traxis.policy_verification import verify_policy
            report = verify_policy(BankDataset.load(args.dataset), args.reconstructors, args.output, guard)
        except Exception as exc:
            parser.exit(1, f"Adaptive checks stopped: {type(exc).__name__}: {exc}\n")
        print(json.dumps({"passed": report["passed"], "evaluation_episodes": report["evaluation_episodes"],
                          "wall_seconds": report["wall_seconds"], "report": str(args.output / "adaptive-report.json")}, indent=2))
        if not report["passed"]:
            parser.exit(1, "Adaptive pilot failed; inspect the report.\n")
    elif args.command == "verify-workflow":
        guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib))
        guard.check(force=True)
        try:
            from traxis.workflow_verification import verify_workflow
            report = verify_workflow(args.output, guard)
        except Exception as exc:
            parser.exit(1, f"Workflow checks stopped: {type(exc).__name__}: {exc}\n")
        print(json.dumps({"passed": report["passed"], "comparison_episodes": report["comparison_episodes"],
                          "ensemble_episodes": report["ensemble_episodes"], "wall_seconds": report["wall_seconds"],
                          "report": str(args.output / "workflow-report.json")}, indent=2))
        if not report["passed"]:
            parser.exit(1, "Workflow pilot failed; inspect the report.\n")
    elif args.command == "verify-models":
        guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib))
        guard.check(force=True)
        try:
            from traxis.model_verification import verify_models
            report = verify_models(args.output, guard)
        except Exception as exc:
            parser.exit(1, f"Model checks stopped: {exc}\n")
        print(json.dumps({"passed": report["passed"], "configurations": len(report["configurations"]),
                          "report": str(args.output / "model-report.json")}, indent=2))
        if not report["passed"]:
            parser.exit(1, "Model debug checks failed; inspect the report.\n")
    elif args.command == "verify":
        from traxis.verification import run_verification
        guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib))
        try:
            report = run_verification(args.output, guard)
        except Exception as exc:
            parser.exit(1, f"Verification stopped: {exc}\nReport: {args.output / 'report.json'}\n")
        print(json.dumps({"passed": report["passed"], "gate_count": len(report["gates"]),
                          "wall_seconds": report["wall_seconds"], "report": str(args.output / "report.json")}, indent=2))
        if not report["passed"]:
            parser.exit(1, "Numerical verification failed; inspect report gates.\n")
