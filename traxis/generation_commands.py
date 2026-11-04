"""Scenario planning and bounded forward-data CLI operations."""

from dataclasses import replace
import json
from pathlib import Path

from traxis.config import Plate
from traxis.forward import ForwardSettings, generate_waveform_bank
from traxis.resources import ResourceGuard
from traxis.scenarios import AcquisitionSettings, SamplingSettings, load_scenario_plan, nuisance_variant, sample_scenarios, save_scenario_plan
from traxis.solver_requests import export_solver_request
from traxis.storage import canonical_hash, write_json


def add_generation_commands(sub):
    plan = sub.add_parser("plan-scenarios", help="write unsolved randomized geometry/nuisance recipes only")
    plan.add_argument("--config", type=Path)
    plan.add_argument("--count", type=int)
    plan.add_argument("--seed", type=int)
    plan.add_argument("--variants", type=int, default=1)
    plan.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export-solver-request", help="export one unsolved SI geometry/trace task")
    export.add_argument("--plan", type=Path, required=True)
    export.add_argument("--index", type=int, default=0)
    export.add_argument("--output", type=Path, required=True)
    generate = sub.add_parser("generate-bank", help="bounded local elastic pilot for one explicit scenario")
    generate.add_argument("--plan", type=Path, required=True)
    generate.add_argument("--index", type=int, default=0)
    generate.add_argument("--settings", type=Path)
    generate.add_argument("--cache-dir", type=Path)
    generate.add_argument("--output", type=Path, required=True)
    for command in (plan, export, generate):
        command.add_argument("--max-seconds", type=float, default=60)
        command.add_argument("--max-rss-mib", type=float, default=512)


def run_generation_command(args, guard: ResourceGuard):
    guard.check(force=True)
    if args.command == "plan-scenarios":
        spec = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
        sampling = SamplingSettings(**spec.get("sampling", {}))
        if args.count is not None:
            sampling = replace(sampling, count=args.count)
        if args.seed is not None:
            sampling = replace(sampling, seed=args.seed)
        if not 1 <= args.variants <= 16 or sampling.count * args.variants > 10000:
            raise ValueError("planning is limited to 10,000 rows and at most sixteen variants per parent")
        cases = sample_scenarios(sampling, Plate(**spec.get("plate", {})), AcquisitionSettings(**spec.get("acquisition", {})))
        all_cases = []
        for scenario in cases:
            guard.check()
            all_cases.append(scenario)
            for variant in range(1, args.variants):
                seed = int(canonical_hash([sampling.seed, scenario.parent_specimen_id, variant])[:16], 16)
                all_cases.append(nuisance_variant(scenario, sampling, seed))
        plan = save_scenario_plan(args.output, tuple(all_cases))
        return {"kind": plan["kind"], "rows": len(all_cases), "parents": plan["parent_count"],
                "split_counts": plan["split_counts"], "fingerprint": plan["fingerprint"], "ready_for_training_data": False}
    scenarios = load_scenario_plan(args.plan)
    if not 0 <= args.index < len(scenarios):
        raise ValueError("scenario index is outside the plan")
    scenario = scenarios[args.index]
    if args.command == "export-solver-request":
        task = export_solver_request(scenario, args.output)
        return {"kind": task["kind"], "records": task["record_count"], "jobs": len(task["jobs"]), "output": str(args.output)}
    settings = ForwardSettings(**json.loads(args.settings.read_text(encoding="utf-8-sig"))) if args.settings else ForwardSettings()
    if args.output.exists() or args.output.with_suffix(".report.json").exists():
        raise FileExistsError("bank or report output already exists")
    bank, report = generate_waveform_bank(scenario, settings, cache_dir=args.cache_dir, guard=guard)
    bank.save(args.output)
    report["resources"] = guard.report()
    write_json(args.output.with_suffix(".report.json"), report)
    return {"output": str(args.output), "records": len(bank.actions), "forward_solves": report["forward_solves"],
            "cache_hits": report["cache_hits"], "validation_status": bank.metadata.validation_status, "ready_for_training_data": False}
