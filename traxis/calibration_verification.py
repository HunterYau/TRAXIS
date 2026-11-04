"""Bounded post-hoc calibration plumbing pilot using existing debug models."""

from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch

from traxis.dataset import BankDataset
from traxis.diagnostic_data import analytic_bank
from traxis.evaluation import freeze_evaluation_plan, run_frozen_evaluation
from traxis.resources import ResourceGuard
from traxis.storage import sha256_file, write_json
from traxis.uncertainty import evaluate_calibration, fit_calibration, freeze_calibration_plan


def verify_calibration(dataset: BankDataset, reconstruction_paths: list[Path], output: Path,
                         guard: ResourceGuard) -> dict:
    if any(e.domain != "simulation" or e.validation_status != "plumbing_only" or e.observable != "arbitrary"
           or e.split not in ("train", "validation", "test") for e in dataset.entries):
        raise ValueError("calibration verification accepts only the analytic software fixture cohort")
    if len(reconstruction_paths) < 2:
        raise ValueError("calibration verification requires existing independent debug ensemble members")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("calibration pilot needs a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(1)
    report = {"kind": "uncertainty_calibration_software_pilot", "debug_only": True, "passed": False,
              "training_updates": 0,
              "note": "Analytic fixtures and existing debug models; declared fixture parents do not establish independent physical specimens, statistical calibration quality or generalization.",
              "source_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    try:
        guard.check(force=True)
        banks = output / "banks"
        banks.mkdir()
        paths = []
        for index, entry in enumerate(dataset.entries):
            path = banks / f"original-{index}.npz"
            dataset.read(entry).save(path)
            paths.append(path)
        for index in range(9):
            guard.check()
            seed = 1000 + index
            rng = np.random.default_rng(seed)
            gain, noise_std = .8 + .04 * index, .005
            fixture = analytic_bank(intact=index % 3 == 0, specimen_id=f"calibration-fixture-scan-{index}",
                                    parent_specimen_id=f"calibration-fixture-parent-{index}",
                                    split="uncertainty_calibration", seed=seed)
            fixture = replace(fixture, waveforms=fixture.waveforms * gain + rng.normal(0, noise_std, fixture.waveforms.shape),
                              baseline=fixture.baseline * gain + rng.normal(0, noise_std, fixture.baseline.shape),
                              metadata=replace(fixture.metadata, settings={**fixture.metadata.settings,
                                                "debug_signal_gain": gain, "debug_noise_std": noise_std}))
            path = banks / f"calibration-{index}.npz"
            fixture.save(path)
            paths.append(path)
        cohort = BankDataset.build(paths, output / "dataset.json", guard)
        specs = ({"kind": "fixed"}, {"kind": "random", "seed": 17})
        replay_plan, calibration_plan = output / "calibration-replay-plan.json", output / "calibration-plan.json"
        freeze_evaluation_plan(replay_plan, cohort, reconstruction_paths, split="uncertainty_calibration",
                               budgets=(2, 4), policy_specs=specs, ensemble=True)
        freeze_calibration_plan(replay_plan, calibration_plan, alpha=.1, std_floor_mm=.05)
        calibration_replay = run_frozen_evaluation(replay_plan, cohort, output / "calibration-replay", guard,
                                                   calibration_plan_path=calibration_plan)
        fitted = fit_calibration(calibration_plan, output / "calibration-replay", output / "calibration.json", guard)
        test_plan = output / "heldout-replay-plan.json"
        freeze_evaluation_plan(test_plan, cohort, reconstruction_paths, split="test", budgets=(2, 4), policy_specs=specs, ensemble=True)
        heldout = run_frozen_evaluation(test_plan, cohort, output / "heldout-replay", guard)
        assessed = evaluate_calibration(output / "calibration.json", output / "heldout-replay", output / "intervals", guard)
        report.update({"calibration_episodes": len(calibration_replay["episodes"]), "heldout_episodes": len(heldout["episodes"]),
                       "calibration_parents": fitted["independent_parents"], "order_statistic_rank": fitted["order_statistic_rank"],
                       "finite_quantile": fitted["finite_quantile"], "quantile": fitted["quantile"],
                       "std_floor_mm": fitted["std_floor_mm"], "nominal_coverage": fitted["nominal_coverage"],
                       "heldout_fixture_parents": assessed["independent_parents"],
                       "empirical_parent_simultaneous_coverage": assessed["empirical_parent_simultaneous_coverage"],
                       "empirical_coverage_meets_nominal": assessed["empirical_coverage_meets_nominal"],
                       "mean_interval_width_mm": float(np.mean([r["mean_interval_width_mm"] for r in assessed["rows"]])),
                       "calibration_sha256": sha256_file(output / "calibration.json"),
                       "assessment_sha256": sha256_file(output / "intervals" / "coverage.json")})
        guard.check(force=True)
        report["passed"] = fitted["finite_quantile"] and fitted["independent_parents"] == 9 and assessed["passed"]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["resources"] = guard.report()
        write_json(output / "calibration-report.json", report)
    return report
