from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from traxis.uncertainty import corrected_quantile, prediction_interval


class ConformalMathTests(unittest.TestCase):
    def test_corrected_rank_exact_leave_one_parent_out_coverage(self):
        # Enumerate all possible held-out ranks; interpolation or a missing
        # n+1 correction would under-cover this independent finite example.
        scores = [1., 2., 3., 4.]
        covered = []
        for index, held_out in enumerate(scores):
            quantile, rank = corrected_quantile(scores[:index] + scores[index + 1:], .25)
            self.assertEqual(rank, 3)
            covered.append(held_out <= quantile)
        self.assertEqual(sum(covered) / len(covered), .75)
        self.assertEqual(corrected_quantile(range(19), .05), (18., 19))
        self.assertEqual(corrected_quantile(range(9), .1), (8., 9))

    def test_small_cohorts_are_explicitly_uninformative_and_inputs_checked(self):
        self.assertEqual(corrected_quantile([1., 2.], .1), (None, 3))
        for scores, alpha in (([], .1), ([float("nan")], .1), ([-1], .1), ([1], 0), ([1], 1)):
            with self.assertRaises(ValueError):
                corrected_quantile(scores, alpha)
        mean, std = np.full((2, 2), 1.), np.zeros((2, 2))
        lower, upper = prediction_interval(mean, std, 3., {"quantile": None, "std_floor_mm": .05})
        np.testing.assert_array_equal(lower, 0)
        np.testing.assert_array_equal(upper, 3)

    def test_floor_units_boundary_clipping_and_exact_ties(self):
        mean, std = np.array([[0., 1.5], [2.9, .4]]), np.zeros((2, 2))
        lower, upper = prediction_interval(mean, std, 3., {"quantile": 2., "std_floor_mm": .1})
        np.testing.assert_allclose(lower, [[0, 1.3], [2.7, .2]], atol=1e-14)
        np.testing.assert_allclose(upper, [[.2, 1.7], [3, .6]], atol=1e-14)
        self.assertTrue(np.all(lower <= mean))
        self.assertTrue(np.all(mean <= upper))


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is required")
class CalibrationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tests.test_dataset import fixture_dataset
        from traxis.evaluation import freeze_evaluation_plan, run_frozen_evaluation
        from traxis.models import ModelConfig, SignalScale, build_model, save_model
        from traxis.uncertainty import freeze_calibration_plan, fit_calibration
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.dataset = fixture_dataset(cls.root, splits=("train", "validation", "uncertainty_calibration",
                                                         "uncertainty_calibration", "uncertainty_calibration", "test"))
        cls.models = []
        scale = SignalScale.fit([cls.dataset.read(cls.dataset.select("train")[0])])
        for seed in (17, 29):
            torch.manual_seed(seed)
            model = build_model(ModelConfig(width=4, embedding=16))
            path = cls.root / f"untrained-{seed}.pt"
            save_model(path, model, scale, {"mode": "debug", "seed": seed,
                                           "fit_parents": ["parent-0"], "validation_parents": ["parent-1"]})
            cls.models.append(path)
        cls.replay_plan = cls.root / "replay-plan.json"
        freeze_evaluation_plan(cls.replay_plan, cls.dataset, cls.models, split="uncertainty_calibration",
                               budgets=(2, 4), ensemble=True)
        cls.calibration_plan = cls.root / "calibration-plan.json"
        freeze_calibration_plan(cls.replay_plan, cls.calibration_plan, alpha=.25)
        cls.replay = cls.root / "calibration-replay"
        run_frozen_evaluation(cls.replay_plan, cls.dataset, cls.replay, calibration_plan_path=cls.calibration_plan)
        cls.calibration_path = cls.root / "calibrated.json"
        cls.calibration = fit_calibration(cls.calibration_plan, cls.replay, cls.calibration_path)
        cls.test_plan = cls.root / "test-plan.json"
        freeze_evaluation_plan(cls.test_plan, cls.dataset, cls.models, split="test", budgets=(2, 4), ensemble=True)
        cls.test_replay = cls.root / "test-replay"
        run_frozen_evaluation(cls.test_plan, cls.dataset, cls.test_replay)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_reserved_parents_never_enter_model_training_or_selection(self):
        from traxis.evaluation import validate_evaluation_plan
        from traxis.training import TrainConfig, TrainingSession
        from traxis.models import ModelConfig
        session = TrainingSession(self.dataset, ModelConfig(width=4, embedding=16), TrainConfig(max_steps=1))
        self.assertEqual([e.parent_specimen_id for e in session.fit_entries], ["parent-0"])
        self.assertEqual([e.parent_specimen_id for e in session.validation_entries], ["parent-1"])
        plan = json.loads(self.replay_plan.read_text())
        exposed = {"fit_parents": ["parent-0"], "validation_parents": ["parent-2"]}
        with self.assertRaisesRegex(ValueError, "held-out parent"):
            validate_evaluation_plan(plan, self.dataset, [exposed, exposed])
        self.assertEqual(self.calibration["independent_parents"], 3)
        self.assertEqual(self.calibration["order_statistic_rank"], 3)
        self.assertTrue(self.calibration["finite_quantile"])
        self.assertTrue(self.calibration["debug_only"])

    def test_heldout_cli_assessment_emits_bounded_interval_maps(self):
        from traxis.cli import main
        output = self.root / "assessment"
        with patch("sys.argv", ["traxis", "assess-calibration", "--calibration", str(self.calibration_path),
                                "--evaluation", str(self.test_replay), "--output", str(output)]):
            main()
        report = json.loads((output / "coverage.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["independent_parents"], 1)
        self.assertEqual(len(report["rows"]), 3)
        with np.load(output / "case-00000-intervals.npz", allow_pickle=False) as maps:
            self.assertTrue(np.all(maps["lower_2_mm"] <= maps["upper_2_mm"]))
            self.assertGreaterEqual(maps["lower_2_mm"].min(), 0)
            self.assertLessEqual(maps["upper_2_mm"].max(), 3)

    def test_model_selection_data_and_research_mode_cannot_be_used_for_fixture_calibration(self):
        from traxis.uncertainty import freeze_calibration_plan
        from traxis.evaluation import run_frozen_evaluation
        with self.assertRaisesRegex(ValueError, "dedicated"):
            freeze_calibration_plan(self.test_plan, self.root / "bad-role.json")
        research_plan = self.root / "research-plan.json"
        freeze_calibration_plan(self.replay_plan, research_plan, alpha=.25, mode="research")
        with self.assertRaisesRegex(ValueError, "rejects debug"):
            run_frozen_evaluation(self.replay_plan, self.dataset, self.root / "bad-research",
                                   calibration_plan_path=research_plan)

    def test_protocol_change_and_reused_calibration_parent_fail(self):
        from traxis.evaluation import freeze_evaluation_plan, run_frozen_evaluation
        from traxis.uncertainty import evaluate_calibration
        altered_plan = self.root / "altered-budget.json"
        freeze_evaluation_plan(altered_plan, self.dataset, self.models, split="test", budgets=(1, 4), ensemble=True)
        altered_replay = self.root / "altered-replay"
        run_frozen_evaluation(altered_plan, self.dataset, altered_replay)
        with self.assertRaisesRegex(ValueError, "changed"):
            evaluate_calibration(self.calibration_path, altered_replay, self.root / "bad-protocol")
        # A different manifest cannot hide re-use of a reserved parent by
        # relabeling that parent's scan as a new final-test entry.
        from traxis.dataset import BankDataset
        reused = self.root / "reused"
        reused.mkdir()
        bank = self.dataset.read(self.dataset.select("uncertainty_calibration")[0])
        replace(bank, metadata=replace(bank.metadata, split="test")).save(reused / "reused.npz")
        dataset = BankDataset.build([reused / "reused.npz"], reused / "dataset.json")
        freeze_evaluation_plan(reused / "plan.json", dataset, self.models, split="test", budgets=(2, 4), ensemble=True)
        run_frozen_evaluation(reused / "plan.json", dataset, reused / "replay")
        with self.assertRaisesRegex(ValueError, "cannot be reused"):
            evaluate_calibration(self.calibration_path, reused / "replay", self.root / "bad-reuse")

    def test_repeated_scans_do_not_inflate_sample_count(self):
        from traxis.uncertainty import _cohort_structure
        report = json.loads((self.replay / "evaluation.json").read_text())
        report["episodes"].append(report["episodes"][0])
        with self.assertRaisesRegex(ValueError, "not independent"):
            _cohort_structure(report)

    def test_calibration_replay_requires_predeclared_settings_and_rejects_refitting_settings(self):
        from traxis.evaluation import run_frozen_evaluation
        from traxis.uncertainty import fit_calibration, freeze_calibration_plan
        with self.assertRaisesRegex(ValueError, "freeze interval settings"):
            run_frozen_evaluation(self.replay_plan, self.dataset, self.root / "undeclared")
        different = self.root / "different-floor.json"
        freeze_calibration_plan(self.replay_plan, different, alpha=.25, std_floor_mm=.1)
        with self.assertRaisesRegex(ValueError, "not bound before"):
            fit_calibration(different, self.replay, self.root / "post-hoc-floor.json")

    def test_selector_validation_exposure_to_calibration_parents_is_rejected(self):
        from traxis.evaluation import validate_selector_provenance
        from traxis.storage import sha256_file
        plan = json.loads(self.replay_plan.read_text())
        provenance = {"fit_parents": ["parent-0"], "validation_parents": ["parent-2"],
                      "reconstruction_artifacts": [{"sha256": sha256_file(path)} for path in self.models],
                      "training": {"initial_actions": []}}
        with self.assertRaisesRegex(ValueError, "held-out parent"):
            validate_selector_provenance(plan, self.dataset, provenance)
