"""Small end-to-end software pilot; never research training or physical data."""

from dataclasses import asdict
from pathlib import Path
import time

import torch

from traxis.das_baseline import fit_das_baseline
from traxis.dataset import BankDataset
from traxis.diagnostic_data import analytic_bank
from traxis.evaluation import freeze_evaluation_plan, run_frozen_evaluation
from traxis.imaging import DASSettings
from traxis.models import ModelConfig
from traxis.resources import ResourceGuard
from traxis.storage import sha256_file, write_json
from traxis.training import TrainConfig, TrainingSession


def verify_workflow(output: Path, guard: ResourceGuard) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("workflow pilot needs a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(1)
    report = {"kind": "software_workflow_pilot", "debug_only": True, "passed": False,
              "note": "Analytic point-scattering fixtures; no physical accuracy, policy superiority or training convergence claim.",
              "source_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
              "training_runs": []}
    try:
        guard.check(force=True)
        banks = output / "banks"
        banks.mkdir()
        paths = []
        for i, role in enumerate(("train", "train", "validation", "test", "test")):
            fixture = analytic_bank(intact=i % 2 == 1, specimen_id=f"debug-scan-{i}", parent_specimen_id=f"debug-parent-{i}", split=role)
            path = banks / f"scan-{i}.npz"
            fixture.save(path)
            paths.append(path)
        dataset = BankDataset.build(paths, output / "dataset.json", guard)
        report["dataset_fingerprint"] = dataset.fingerprint
        settings = DASSettings({100000.: 2500.}, 25e-6)
        model_paths, direct_paths = [], []
        for architecture, seed, updates in (("direct", 101, 2), ("direct", 202, 1), ("das_unet", 101, 1)):
            cfg = TrainConfig(max_steps=updates, seed=seed, min_records=2, max_records=4)
            config = ModelConfig(architecture, width=8, embedding=32)
            session = TrainingSession(dataset, config, cfg, settings, guard=guard)
            checkpoint = output / f"{architecture}-{seed}-latest.pt"
            session.run(1, checkpoint)
            if updates > 1:
                session = TrainingSession.resume(checkpoint, dataset, guard)
                session.run(updates - 1, checkpoint)
            exported = output / f"{architecture}-{seed}.pt"
            session.export_model(exported)
            record = {"config": asdict(config), "training": asdict(cfg), "updates": session.step_count,
                      "history": session.history, "validation_history": session.validation_history,
                      "parameters": sum(p.numel() for p in session.model.parameters()),
                      "checkpoint_sha256": sha256_file(exported)}
            report["training_runs"].append(record)
            if architecture == "direct":
                direct_paths.append(exported)
            if seed == 101:
                model_paths.append(exported)
            del session
        calibration = fit_das_baseline(dataset, settings, (2, 4), guard=guard)
        baseline_path = output / "das-depth.json"
        calibration.save(baseline_path)
        model_paths.append(baseline_path)
        plan_path = output / "comparison-plan.json"
        freeze_evaluation_plan(plan_path, dataset, model_paths, split="test", budgets=(2, 4),
                               policy_specs=({"kind": "fixed"}, {"kind": "random", "seed": 13}, {"kind": "random", "seed": 29}))
        comparison = run_frozen_evaluation(plan_path, dataset, output / "comparison", guard)
        ensemble_plan = output / "ensemble-plan.json"
        freeze_evaluation_plan(ensemble_plan, dataset, direct_paths, split="test", budgets=(2, 4),
                               policy_specs=({"kind": "fixed"}, {"kind": "random", "seed": 13}), ensemble=True)
        ensemble = run_frozen_evaluation(ensemble_plan, dataset, output / "ensemble", guard)
        # Fixed/random selection is prediction-independent: these actions must
        # be identical across all three reconstruction methods at every budget.
        actions = {}
        for episode in comparison["episodes"]:
            for row in episode["rows"]:
                key = (episode["specimen_id"], str(episode["policy"]), row["records"])
                previous = actions.setdefault(key, row["action_indices"])
                if previous != row["action_indices"]:
                    raise AssertionError("architecture comparison used different observation sets")
        report.update({"comparison_episodes": len(comparison["episodes"]),
                       "ensemble_episodes": len(ensemble["episodes"]),
                       "matched_actions": True, "comparison_summary": comparison["summary"],
                       "ensemble_summary": ensemble["summary"],
                       "ensemble_uncertainty": [{"specimen_id": e["specimen_id"], "policy": e["policy"],
                                                  "rows": [{"records": r["records"], "uncertainty": r["uncertainty"]} for r in e["rows"]]}
                                                 for e in ensemble["episodes"]]})
        report["passed"] = comparison["passed"] and ensemble["passed"] and len(comparison["episodes"]) == 18 and len(ensemble["episodes"]) == 4
        guard.check(force=True)
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["resources"] = guard.report()
        write_json(output / "workflow-report.json", report)
    return report
