"""Bounded adaptive-policy pilot over existing software fixtures."""

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from traxis.adaptive import PolicyConfig
from traxis.dataset import BankDataset
from traxis.evaluation import freeze_evaluation_plan, run_frozen_evaluation
from traxis.policy_training import PolicyTrainingSession, RLTrainConfig
from traxis.resources import ResourceGuard
from traxis.storage import sha256_file, write_json


def verify_policy(dataset: BankDataset, reconstruction_paths: list[Path], output: Path, guard: ResourceGuard):
    if len(dataset.entries) > 8 or not dataset.select("test") or len(dataset.select("test")) > 4:
        raise ValueError("policy pilot needs a small fixture dataset with one to four test specimens")
    if any(e.domain != "simulation" or e.validation_status != "plumbing_only" or e.observable != "arbitrary" for e in dataset.entries):
        raise ValueError("policy verification is restricted to software fixtures, not research or physical holdouts")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("adaptive pilot needs a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"kind": "adaptive_software_pilot", "debug_only": True, "passed": False,
              "note": "Fixture checks only; no policy superiority, stopping, physical accuracy or acquisition-time claim.",
              "source_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    try:
        config, training = PolicyConfig(), RLTrainConfig(max_episodes=2, horizon=3)
        session = PolicyTrainingSession(dataset, reconstruction_paths, config, training, guard)
        checkpoint = output / "latest-policy.pt"
        session.run(1, checkpoint)
        session = PolicyTrainingSession.resume(checkpoint, dataset, guard)
        session.run(1, checkpoint)
        selected = output / "selected-policy.pt"
        session.export_policy(selected)
        plan_path = output / "adaptive-plan.json"
        specs = ({"kind": "fixed"}, {"kind": "random", "seed": 17}, {"kind": "random", "seed": 29},
                 {"kind": "random", "seed": 43}, {"kind": "uncertainty_path"}, {"kind": "rl", "path": str(selected)})
        freeze_evaluation_plan(plan_path, dataset, reconstruction_paths, split="test", budgets=(1, 2, 3),
                               policy_specs=specs, ensemble=True)
        result = run_frozen_evaluation(plan_path, dataset, output / "comparison", guard)
        dense_maps, budgets, policies = {}, set(), set()
        for index, episode in enumerate(result["episodes"]):
            policies.add(episode["policy"]["kind"])
            counts = tuple(row["records"] for row in episode["rows"])
            budgets.add(counts)
            for row in episode["rows"]:
                if len(set(row["action_indices"])) != row["records"]:
                    raise AssertionError("adaptive replay repeated or omitted an acquisition")
            dense_count = episode["rows"][-1]["records"]
            with np.load(output / "comparison" / f"case-{index:05d}" / "maps.npz", allow_pickle=False) as saved:
                dense = saved[f"depth_{dense_count}_mm"]
                parent = episode["specimen_id"]
                if parent in dense_maps:
                    np.testing.assert_allclose(dense, dense_maps[parent], rtol=0, atol=1e-7)
                else:
                    dense_maps[parent] = dense.copy()
        report.update({"policy_config": asdict(config), "training_config": asdict(training),
                       "parameters": sum(p.numel() for p in session.network.parameters()),
                       "training_history": session.history, "validation_history": session.validation_history,
                       "provenance": session.provenance(), "selected_policy_sha256": sha256_file(selected),
                       "evaluation_episodes": len(result["episodes"]), "policy_kinds": sorted(policies),
                       "record_budgets": [list(b) for b in sorted(budgets)], "common_dense_prediction": True,
                       "no_repeated_actions": True, "summary": result["summary"]})
        expected = len(dataset.select("test")) * len(specs)
        report["passed"] = result["passed"] and len(result["episodes"]) == expected and len(policies) == 4
        guard.check(force=True)
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["resources"] = guard.report()
        write_json(output / "adaptive-report.json", report)
    return report
