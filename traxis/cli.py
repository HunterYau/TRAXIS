import argparse
from dataclasses import asdict
import json
from pathlib import Path

from traxis.acquisition import action_catalog
from traxis.geometry import estimate_mesh
from traxis.resources import ResourceGuard, ResourceLimits, snapshot


def main() -> None:
    from traxis.experiment_commands import add_experiment_commands
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
    add_experiment_commands(sub)
    args = parser.parse_args()
    if args.command in ("index-data", "train", "train-policy", "calibrate-das", "freeze-evaluation", "evaluate"):
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
