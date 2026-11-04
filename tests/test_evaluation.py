import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.fixtures import analytic_bank
from tests.test_dataset import fixture_dataset
from traxis.policies import FixedPolicy, RandomPolicy

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from traxis.evaluation import freeze_evaluation_plan, replay_episode, run_frozen_evaluation, summarize_episodes, validate_evaluation_plan
    from traxis.reconstruction import Reconstruction
    from traxis.training import TrainConfig, TrainingSession
    from traxis.models import ModelConfig


@unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is required")
class EvaluationTests(unittest.TestCase):
    def test_equal_budgets_initial_state_and_no_truth_in_policy(self):
        bank = analytic_bank()
        class Predictor:
            def predict(self, observation):
                return Reconstruction(np.full((128, 128), 1 / (1 + observation.count)))
        class AuditedPolicy:
            def __init__(self):
                self.delegate, self.counts = FixedPolicy(), []
            def choose(self, observation, reconstruction):
                self.asserted = not hasattr(observation, "depth_mm") and not observation.waveforms[~observation.observed].any()
                if not self.asserted:
                    raise AssertionError("hidden state leaked")
                self.counts.append(observation.count)
                return self.delegate.choose(observation, reconstruction)
        for policy in (AuditedPolicy(), RandomPolicy(12)):
            result = replay_episode(bank, Predictor(), policy, (2, 4), initial_actions=(0,))
            self.assertEqual([row["records"] for row in result.report["rows"]], [2, 4, 28])
            self.assertTrue(all(row["action_indices"][0] == 0 for row in result.report["rows"]))
            self.assertIsNone(result.report["hardware_acquisition_seconds"])
            self.assertEqual(len(set(result.report["rows"][-1]["action_indices"])), 28)

    def test_near_zero_dense_error_has_no_misleading_ratio(self):
        bank = analytic_bank(intact=True)
        class Zeros:
            def predict(self, observation):
                return Reconstruction(np.zeros((128, 128)))
        result = replay_episode(bank, Zeros(), FixedPolicy(), (0, 1))
        self.assertIsNone(result.report["rows"][0]["relative_mae_to_dense"])
        with tempfile.TemporaryDirectory() as directory:
            result.save(Path(directory) / "result")
            with self.assertRaises(FileExistsError):
                result.save(Path(directory) / "result")

    def test_frozen_plan_detects_model_changes_and_heldout_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = fixture_dataset(root)
            artifact = root / "model.pt"
            artifact.write_bytes(b"fingerprint fixture")
            plan = freeze_evaluation_plan(root / "plan.json", dataset, [artifact], split="test", budgets=(1, 4))
            valid = {"fit_parents": ["parent-0"], "validation_parents": ["parent-1"]}
            validate_evaluation_plan(plan, dataset, [valid])
            with self.assertRaisesRegex(ValueError, "held-out"):
                validate_evaluation_plan(plan, dataset, [{**valid, "fit_parents": ["parent-2"]}])
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "model changed"):
                validate_evaluation_plan(plan, dataset, [valid])

    def test_run_actual_bound_checkpoint_and_preserve_each_case(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = fixture_dataset(root)
            session = TrainingSession(dataset, ModelConfig(width=4, embedding=16), TrainConfig(max_steps=1))
            session.run(1)
            model_path = root / "selected.pt"
            session.export_model(model_path)
            plan_path = root / "plan.json"
            freeze_evaluation_plan(plan_path, dataset, [model_path], split="test", budgets=(1, 4),
                                   policy_specs=({"kind": "fixed"}, {"kind": "random", "seed": 12}))
            report = run_frozen_evaluation(plan_path, dataset, root / "evaluation")
            self.assertTrue(report["passed"])
            self.assertTrue(report["debug_only"])
            self.assertEqual(len(report["episodes"]), 2)
            self.assertEqual(len(list((root / "evaluation").glob("case-*/maps.npz"))), 2)

    def test_repeats_and_random_seeds_do_not_inflate_independent_parent_count(self):
        def episode(parent, scan, value):
            metrics = {key: value for key in ("mae_mm", "rmse_mm", "defect_region_mae_mm", "dice", "iou", "centroid_error_mm", "false_positive_area_mm2", "loss_volume_error_mm3")}
            return {"predictor": "A", "policy": {"kind": "random"}, "parent_specimen_id": parent,
                    "specimen_id": scan, "rows": [{"records": 4, "metrics": metrics}]}
        values = [episode("p1", "scan1", 1), episode("p1", "scan1", 3), episode("p1", "scan2", 4), episode("p2", "scan3", 8)]
        summary = summarize_episodes(values)[0]
        self.assertEqual(summary["per_parent"]["p1"]["mae_mm"], 3)
        self.assertEqual(summary["metrics"]["mae_mm"]["mean"], 5.5)
        self.assertEqual(summary["metrics"]["mae_mm"]["independent_parents"], 2)
