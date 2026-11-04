from dataclasses import replace
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_dataset import fixture_dataset

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from traxis.adaptive import ActorCriticPolicy, PolicyConfig
    from traxis.evaluation import freeze_evaluation_plan, run_frozen_evaluation, validate_evaluation_plan, validate_selector_provenance
    from traxis.models import ModelConfig
    from traxis.policy_training import PolicyTrainingSession, RLTrainConfig
    from traxis.resources import ResourceLimitError
    from traxis.training import TrainConfig, TrainingSession


@unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is required")
class PolicyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.dataset = fixture_dataset(cls.root, ("train", "train", "validation", "test", "test"))
        cls.paths = []
        for seed in (11, 22):
            model = TrainingSession(cls.dataset, ModelConfig(width=4, embedding=16), TrainConfig(max_steps=1, seed=seed))
            model.run(1)
            path = cls.root / f"reconstructor-{seed}.pt"
            model.export_model(path)
            cls.paths.append(path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def session(self, **changes):
        return PolicyTrainingSession(self.dataset, self.paths, PolicyConfig(width=4, embedding=16),
                                     RLTrainConfig(max_episodes=3, horizon=2, **changes))

    def test_on_policy_update_is_finite_masked_and_leaves_reconstructor_frozen(self):
        session = self.session()
        before = {k: v.clone() for k, v in session.network.state_dict().items()}
        with patch.object(session.predictor, "predict", wraps=session.predictor.predict) as predict:
            row = session.train_episode()
            self.assertEqual(predict.call_count, 3)  # Initial state and two chosen records only.
        self.assertEqual(len(set(row["action_indices"])), 2)
        self.assertTrue(np.isfinite(row["loss"]))
        self.assertGreater(row["gradient_norm_before_clip"], 0)
        self.assertTrue(any(not torch.equal(v, session.network.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(any(not torch.equal(v, session.network.state_dict()[k]) for k, v in before.items() if k.startswith("actor.")))
        for member in session.predictor.members:
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in member.model.parameters()))
        expected = (row["initial_balanced_error_mm"] - row["final_balanced_error_mm"]) / 3 - 2 * session.config.record_cost
        self.assertAlmostEqual(sum(row["rewards"]), expected, places=12)

    def test_resume_restores_sampled_actions_optimizer_and_weights_exactly(self):
        uninterrupted = self.session()
        uninterrupted.run(3)
        split = self.session()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy-state.pt"
            split.run(1, checkpoint)
            resumed = PolicyTrainingSession.resume(checkpoint, self.dataset)
            resumed.run(2, checkpoint)
            self.assertEqual(uninterrupted.history, resumed.history)
            self.assertEqual(uninterrupted.validation_history, resumed.validation_history)
            for key, value in uninterrupted.network.state_dict().items():
                torch.testing.assert_close(value, resumed.network.state_dict()[key], rtol=0, atol=0)

    def test_interrupted_policy_validation_preserves_completed_episode(self):
        original = self.session()
        original.run(2)
        stopped = self.session()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            with patch.object(stopped, "validate", side_effect=ResourceLimitError("test interruption")):
                with self.assertRaises(ResourceLimitError):
                    stopped.run(1, path)
            restored = PolicyTrainingSession.resume(path, self.dataset)
            self.assertEqual(restored.episode_count, 1)
            self.assertTrue(restored.pending_validation)
            restored.run(1, path)
            self.assertEqual(original.history, restored.history)
            self.assertEqual(original.validation_history, restored.validation_history)

    def test_research_and_uncertainty_gates(self):
        with self.assertRaises(ValueError):
            RLTrainConfig(max_episodes=5)
        with self.assertRaises(ValueError):
            RLTrainConfig(horizon=9)
        with self.assertRaisesRegex(ValueError, "research"):
            PolicyTrainingSession(self.dataset, self.paths, PolicyConfig(), RLTrainConfig(mode="research"))
        with self.assertRaisesRegex(ValueError, "ensemble"):
            PolicyTrainingSession(self.dataset, self.paths[:1], PolicyConfig(), RLTrainConfig())
        ablation = PolicyTrainingSession(self.dataset, self.paths[:1], PolicyConfig(width=4, embedding=16, require_uncertainty=False),
                                         RLTrainConfig(horizon=1))
        self.assertTrue(np.isfinite(ablation.train_episode()["loss"]))

    def test_bound_selector_frozen_replay_and_lineage_rejection(self):
        session = self.session()
        session.run(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy_path, plan_path = root / "selected-policy.pt", root / "plan.json"
            session.export_policy(policy_path)
            plan = freeze_evaluation_plan(plan_path, self.dataset, self.paths, split="test", budgets=(1, 2),
                                          policy_specs=({"kind": "fixed"}, {"kind": "random", "seed": 17},
                                                        {"kind": "uncertainty_path"}, {"kind": "rl", "path": str(policy_path)}), ensemble=True)
            result = run_frozen_evaluation(plan_path, self.dataset, root / "evaluation")
            self.assertTrue(result["passed"])
            self.assertTrue(result["debug_only"])
            self.assertEqual(len(result["episodes"]), 8)
            provenance = ActorCriticPolicy.load(policy_path).provenance
            with self.assertRaisesRegex(ValueError, "held-out"):
                validate_selector_provenance(plan, self.dataset, {**provenance, "fit_parents": ["parent-4"]})
            with self.assertRaisesRegex(ValueError, "different reconstruction"):
                validate_selector_provenance({**plan, "models": plan["models"][:1]}, self.dataset, provenance)
            with self.assertRaisesRegex(ValueError, "initial"):
                validate_selector_provenance({**plan, "initial_actions": [0]}, self.dataset, provenance)
            policy_path.write_bytes(policy_path.read_bytes() + b"changed-selector")
            with self.assertRaisesRegex(ValueError, "selector changed"):
                validate_evaluation_plan(plan, self.dataset, session.reconstruction_provenances)

    def test_policy_commands_train_resume_and_freeze_bound_artifacts(self):
        from traxis.cli import main
        def invoke(args):
            captured = io.StringIO()
            with patch("sys.argv", ["traxis", *map(str, args)]), contextlib.redirect_stdout(captured):
                main()
            return json.loads(captured.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "policy-config.json"
            config.write_text(json.dumps({"policy": {"width": 4, "embedding": 16},
                                           "training": {"max_episodes": 2, "horizon": 1}}))
            out = root / "run"
            first = invoke(["train-policy", "--dataset", self.root / "dataset.json", "--config", config,
                            "--reconstructors", *self.paths, "--episodes", 1, "--output", out])
            self.assertEqual(first["episodes_total"], 1)
            resumed = invoke(["train-policy", "--dataset", self.root / "dataset.json", "--resume", out / "latest-policy.pt",
                              "--episodes", 1, "--output", out])
            self.assertEqual(resumed["episodes_total"], 2)
            plan_path = root / "plan.json"
            invoke(["freeze-evaluation", "--dataset", self.root / "dataset.json", "--models", *self.paths,
                    "--split", "test", "--budgets", 1, 2, "--ensemble", "--uncertainty-path", "--rl-policies", resumed["policy"],
                    "--output", plan_path])
            plan = json.loads(plan_path.read_text())
            self.assertTrue(any(spec["kind"] == "rl" and "sha256" in spec for spec in plan["policy_specs"]))
            self.assertIn("inference_source_hash", plan)
