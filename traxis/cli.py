import argparse
from dataclasses import asdict
import json
from pathlib import Path

from traxis.acquisition import action_catalog
from traxis.geometry import estimate_mesh
from traxis.resources import ResourceGuard, ResourceLimits, snapshot


def main() -> None:
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
    args = parser.parse_args()
    if args.command == "preflight":
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
