from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_dataset import fixture_dataset
from traxis.imaging import DASSettings

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from traxis.models import ModelConfig, load_model, observation_batch
    from traxis.bank import ReplayEnvironment
    from traxis.training import TrainConfig, TrainingSession, fit_signal_scale
    from traxis.reconstruction import EnsembleReconstructor, NeuralReconstructor
    from traxis.resources import ResourceLimitError


@unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is required")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_resume_restores_optimizer_subsets_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = fixture_dataset(root)
            cfg = TrainConfig(max_steps=3, min_records=1, max_records=6)
            model = ModelConfig(width=4, embedding=16)
            uninterrupted = TrainingSession(dataset, model, cfg)
            uninterrupted.run(3)
            interrupted = TrainingSession(dataset, model, cfg)
            checkpoint = root / "latest.pt"
            interrupted.run(1, checkpoint)
            resumed = TrainingSession.resume(checkpoint, dataset)
            resumed.run(2, checkpoint)
            self.assertEqual(uninterrupted.history, resumed.history)
            self.assertEqual(uninterrupted.validation_history, resumed.validation_history)
            for name, value in uninterrupted.model.state_dict().items():
                torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
            resumed.export_model(root / "selected.pt")
            selected, scale, provenance = load_model(root / "selected.pt")
            self.assertEqual(scale, resumed.scale)
            self.assertEqual(provenance["weights_step"], resumed.best_step)
            self.assertEqual(provenance["fit_parents"], ["parent-0"])
            self.assertEqual(provenance["validation_parents"], ["parent-1"])

    def test_both_architectures_use_partial_subsets_and_compact_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = fixture_dataset(directory)
            for architecture in ("direct", "das_unet"):
                session = TrainingSession(dataset, ModelConfig(architecture, width=4, embedding=16),
                                          TrainConfig(max_steps=1, min_records=2, max_records=2),
                                          DASSettings({100000.: 2500.}, 25e-6))
                row = session.train_step()
                self.assertEqual(row["records"], 2)
                self.assertTrue(np.isfinite(row["loss_mm2"]))
                bank = dataset.read(dataset.select("train")[0])
                env = ReplayEnvironment(bank)
                state = env.reveal(0)
                full, compact = observation_batch([state], session.scale), observation_batch([state], session.scale, compact=True)
                self.assertEqual(compact.signals.shape[1], 1)
                session.model.eval()
                if architecture == "direct":
                    with torch.inference_mode():
                        torch.testing.assert_close(session.model(full), session.model(compact), rtol=1e-5, atol=1e-5)

    def test_debug_cap_research_gate_and_foreign_checkpoint_protection(self):
        with self.assertRaises(ValueError):
            TrainConfig(max_steps=9)
        with tempfile.TemporaryDirectory() as directory:
            dataset = fixture_dataset(directory)
            model = ModelConfig(width=4, embedding=16)
            with self.assertRaisesRegex(ValueError, "plumbing"):
                TrainingSession(dataset, model, TrainConfig(mode="research"))
            first = TrainingSession(dataset, model, TrainConfig())
            path = Path(directory) / "state.pt"
            first.run(1, path)
            second = TrainingSession(dataset, model, TrainConfig())
            with self.assertRaises(FileExistsError):
                second.run(1, path)

    def test_adaptation_freezes_scale_and_excludes_final_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Build pretraining data in a separate directory; keep distinct parent IDs.
            sim_root = root / "sim"
            sim_root.mkdir()
            sim = fixture_dataset(sim_root)
            session = TrainingSession(sim, ModelConfig(width=4, embedding=16), TrainConfig(max_steps=1))
            session.run(1)
            session.export_model(root / "base.pt")
            physical_root = root / "physical"
            physical_root.mkdir()
            physical = fixture_dataset(physical_root, ("adaptation", "physical_holdout"))
            # Parent labels overlap with the simulated names deliberately: the
            # initial selected model used parent-1 for validation, so reject it.
            with self.assertRaisesRegex(ValueError, "holdout"):
                TrainingSession.adapt(root / "base.pt", physical, TrainConfig(phase="adaptation"))
            with self.assertRaises(ValueError):
                TrainingSession(physical, ModelConfig(width=4, embedding=16), TrainConfig())

    def test_independent_ensembles_and_empty_states(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = fixture_dataset(directory)
            first = TrainingSession(dataset, ModelConfig(width=4, embedding=16), TrainConfig(seed=1))
            second = TrainingSession(dataset, ModelConfig(width=4, embedding=16), TrainConfig(seed=2))
            a, b = NeuralReconstructor(first.model, first.scale), NeuralReconstructor(second.model, second.scale)
            ensemble = EnsembleReconstructor([a, b])
            state = ReplayEnvironment(dataset.read(dataset.select("train")[0])).state()
            result = ensemble.predict(state)
            self.assertTrue(np.isfinite(result.std_mm).all())
            self.assertGreater(float(result.std_mm.mean()), 0)
            with self.assertRaises(ValueError):
                EnsembleReconstructor([a, a])

    def test_interrupted_validation_keeps_completed_update_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = fixture_dataset(root)
            cfg, model = TrainConfig(max_steps=2), ModelConfig(width=4, embedding=16)
            original = TrainingSession(dataset, model, cfg)
            original.run(2)
            stopped = TrainingSession(dataset, model, cfg)
            checkpoint = root / "interrupted.pt"
            with patch.object(stopped, "validate", side_effect=ResourceLimitError("test interruption")):
                with self.assertRaises(ResourceLimitError):
                    stopped.run(1, checkpoint)
            resumed = TrainingSession.resume(checkpoint, dataset)
            self.assertEqual(resumed.step_count, 1)
            self.assertTrue(resumed.pending_validation)
            resumed.run(1, checkpoint)
            self.assertEqual(resumed.validation_history, original.validation_history)
            for key, value in resumed.model.state_dict().items():
                torch.testing.assert_close(value, original.model.state_dict()[key], rtol=0, atol=0)

    def test_two_plate_adaptation_excludes_three_final_plates_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sim_root, real_root = root / "sim", root / "real"
            sim_root.mkdir()
            real_root.mkdir()
            sim = fixture_dataset(sim_root)
            base = TrainingSession(sim, ModelConfig(width=4, embedding=16), TrainConfig(max_steps=1))
            base.run(1)
            base.export_model(root / "base.pt")
            physical = fixture_dataset(real_root, ("adaptation", "adaptation", "physical_holdout", "physical_holdout", "physical_holdout"), prefix="real-")
            config = TrainConfig(phase="adaptation", max_steps=2)
            adapted = TrainingSession.adapt(root / "base.pt", physical, config)
            self.assertEqual(adapted.scale, base.scale)
            checkpoint = root / "adapt.pt"
            adapted.run(1, checkpoint)
            resumed = TrainingSession.resume(checkpoint, physical)
            resumed.run(1, checkpoint)
            self.assertTrue(all(r["specimen_id"] in ("real-scan-0", "real-scan-1") for r in resumed.history))
            self.assertTrue(resumed.parent_checkpoint_sha256)
            self.assertEqual(resumed.scale, base.scale)
            self.assertEqual(set(resumed.provenance()["fit_parents"]), {"parent-0", "real-parent-0", "real-parent-1"})
