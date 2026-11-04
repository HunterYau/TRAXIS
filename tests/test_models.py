from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.fixtures import analytic_bank
from traxis.bank import ReplayEnvironment

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from traxis.models import (ModelConfig, ObservationBatch, SignalScale, build_model,
                               load_model, observation_batch, reconstruction_loss, save_model)


@unittest.skipUnless(HAS_TORCH, "install optional CPU PyTorch to check reconstruction candidates")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bank = analytic_bank()
        cls.scale = SignalScale.fit([cls.bank])
        env = ReplayEnvironment(cls.bank)
        for index in (0, 3, 12):
            cls.state = env.reveal(index)
        cls.batch = observation_batch([cls.state], cls.scale)

    def test_all_resolution_and_decoder_variants_forward_backward(self):
        for architecture in ("direct", "das_unet"):
            for size in (128, 256):
                for extra in (0, 1):
                    with self.subTest(architecture=architecture, size=size, extra=extra):
                        torch.manual_seed(12)
                        model = build_model(ModelConfig(architecture, size, width=4, embedding=16, extra_decoder_blocks=extra))
                        output = model(self.batch) if architecture == "direct" else model(self.batch, torch.zeros(1, 2, size, size))
                        self.assertEqual(output.shape, (1, 1, size, size))
                        self.assertTrue(torch.isfinite(output).all())
                        self.assertTrue(((output >= 0) & (output <= 3)).all())
                        loss = reconstruction_loss(output, torch.zeros_like(output))
                        loss.backward()
                        gradients = [p.grad for p in model.parameters() if p.grad is not None]
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertTrue(any(g.abs().sum() > 0 for g in gradients))

    def test_hidden_values_cannot_change_predictions_and_empty_state_is_finite(self):
        dirty = self.batch.signals.clone()
        dirty_context = self.batch.context.clone()
        dirty[~self.batch.observed] = float("nan")
        dirty_context[~self.batch.observed] = float("nan")
        hidden = ObservationBatch(dirty, dirty_context, self.batch.observed)
        empty = observation_batch([ReplayEnvironment(self.bank).state()], self.scale)
        for architecture in ("direct", "das_unet"):
            model = build_model(ModelConfig(architecture, width=4, embedding=16)).eval()
            args = () if architecture == "direct" else (torch.zeros(1, 2, 128, 128),)
            with torch.inference_mode():
                baseline, changed = model(self.batch, *args), model(hidden, *args)
                torch.testing.assert_close(baseline, changed, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(model(empty, *args)).all())

    def test_record_permutation_invariance_and_observed_signal_sensitivity(self):
        permutation = torch.randperm(self.batch.observed.shape[1])
        shuffled = ObservationBatch(self.batch.signals[:, permutation], self.batch.context[:, permutation], self.batch.observed[:, permutation])
        changed = ObservationBatch(self.batch.signals * 0.1, self.batch.context, self.batch.observed)
        for architecture in ("direct", "das_unet"):
            model = build_model(ModelConfig(architecture, width=4, embedding=16)).eval()
            args = () if architecture == "direct" else (torch.zeros(1, 2, 128, 128),)
            with torch.inference_mode():
                output = model(self.batch, *args)
                torch.testing.assert_close(output, model(shuffled, *args), rtol=2e-5, atol=2e-5)
                self.assertGreater(float(abs(output - model(changed, *args)).max()), 1e-6)

    def test_checkpoint_roundtrip(self):
        for architecture in ("direct", "das_unet"):
            model = build_model(ModelConfig(architecture, width=4, embedding=16)).eval()
            args = () if architecture == "direct" else (torch.zeros(1, 2, 128, 128),)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "weights.pt"
                save_model(path, model, self.scale, {"kind": "untrained regression fixture"})
                restored, scale, provenance = load_model(path)
                with torch.inference_mode():
                    torch.testing.assert_close(model(self.batch, *args), restored(self.batch, *args), rtol=0, atol=0)
                self.assertEqual(scale, self.scale)
                self.assertIn("kind", provenance)
                with self.assertRaises(FileExistsError):
                    save_model(path, model, self.scale, {})

    def test_scaling_rejects_test_data_and_mismatched_units(self):
        with self.assertRaises(ValueError):
            SignalScale.fit([analytic_bank(split="test")])
        with self.assertRaises(ValueError):
            observation_batch([replace(self.state, observable="voltage_v")], self.scale)

    def test_real_zero_waveform_is_distinct_from_missing_record(self):
        env = ReplayEnvironment(analytic_bank(intact=True))
        state = env.reveal(0)
        state = replace(state, waveforms=np.zeros_like(state.waveforms), baseline=np.zeros_like(state.baseline))
        batch = observation_batch([state], self.scale)
        self.assertTrue(batch.observed[0, 0])
        self.assertFalse(batch.signals.any())

    def test_batched_states_and_gradients_at_empty_observation(self):
        empty = ReplayEnvironment(self.bank).state()
        batch = observation_batch([empty, self.state], self.scale)
        model = build_model(ModelConfig(width=4, embedding=16))
        output = model(batch)
        reconstruction_loss(output, torch.zeros_like(output)).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
