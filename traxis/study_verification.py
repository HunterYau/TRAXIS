"""One debug update per study member, followed by isolated replay checks."""

import json
from pathlib import Path
import time

from traxis.dataset import BankDataset
from traxis.evaluation import run_frozen_evaluation
from traxis.resources import ResourceGuard
from traxis.storage import canonical_hash, sha256_file, write_json
from traxis.studies import advance_study, create_study, freeze_study_comparisons


def verify_study(dataset_path: Path, configuration: Path, output: Path, guard: ResourceGuard) -> dict:
    dataset = BankDataset.load(dataset_path)
    if any(e.domain != "simulation" or e.observable != "arbitrary" or e.validation_status != "plumbing_only" for e in dataset.entries):
        raise ValueError("study verification accepts only labeled software fixtures")
    if Path(configuration).stat().st_size > 1024**2:
        raise MemoryError("debug study configuration exceeds 1 MiB")
    specification = json.loads(Path(configuration).read_text(encoding="utf-8-sig"))
    if specification["training"]["mode"] != "debug" or specification["training"]["max_steps"] != 1:
        raise ValueError("study verification permits exactly one debug update per neural job")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("study verification needs a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"kind": "matched_study_software_pilot", "passed": False, "debug_only": True,
              "source_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
              "configuration_sha256": sha256_file(configuration),
              "note": "One update per member on analytic fixtures. No research training, architectural superiority, policy convergence or physical accuracy claim. Only predeclared representatives of the controlled comparisons are replayed."}
    try:
        guard.check(force=True)
        plan = create_study(dataset_path, specification, output / "study", guard)
        neural = [job for job in plan["jobs"] if job["kind"] == "neural"]
        if len(neural) != 16 or len(plan["jobs"]) != 20 or len(plan["controlled_pairs"]) != 24:
            raise ValueError("verification expects eight architecture configurations, two paired trials and four DAS fits")
        progress = advance_study(output / "study" / "study.json", max_new_updates=16, baseline_fits=4, guard=guard)
        frozen = freeze_study_comparisons(output / "study" / "study.json", output / "comparison-plans",
                                          split="validation", budgets=(2, 4), random_seeds=(17,), guard=guard)
        report.update({"study_fingerprint": plan["fingerprint"], "neural_jobs": len(neural),
                       "new_updates": progress["new_updates"], "baseline_fits": progress["baseline_fits"],
                       "controlled_pairs": len(frozen["controlled_pairs"]), "ensemble_plans": len(frozen["ensembles"]),
                       "predictors": frozen["predictors"], "replays": []})
        for factor in ("architecture", "output_size", "extra_decoder_blocks"):
            comparison = next(pair for pair in frozen["controlled_pairs"] if pair["factor"] == factor)
            path = output / "comparison-plans" / comparison["plan"]
            replay = run_frozen_evaluation(path, dataset, output / "replay" / factor, guard)
            actions = {}
            for episode in replay["episodes"]:
                for row in episode["rows"]:
                    key = (episode["specimen_id"], canonical_hash(episode["policy"]), row["records"])
                    if actions.setdefault(key, row["action_indices"]) != row["action_indices"]:
                        raise AssertionError("controlled replay used different acquired records")
            report["replays"].append({"factor": factor, "plan": comparison["plan"], "predictors": comparison["predictors"],
                                      "episodes": len(replay["episodes"]), "matched_actions": True, "passed": replay["passed"]})
        ensemble = run_frozen_evaluation(output / "comparison-plans" / frozen["ensembles"][0]["plan"], dataset,
                                          output / "replay" / "ensemble", guard)
        report["replays"].append({"factor": "independent_ensemble", "episodes": len(ensemble["episodes"]), "passed": ensemble["passed"]})
        report["replay_episodes"] = sum(item["episodes"] for item in report["replays"])
        report["passed"] = progress["new_updates"] == 16 and progress["baseline_fits"] == 4 and all(item["passed"] for item in report["replays"])
        guard.check(force=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["resources"] = guard.report()
        write_json(output / "study-report.json", report)
    return report
