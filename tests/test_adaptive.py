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
    from torch.distributions import Categorical
    from traxis.adaptive import ActorCriticPolicy, MaskedActorCritic, PolicyConfig, UncertaintyPathPolicy, spatial_features
    from traxis.models import SignalScale
    from traxis.policy_training import RLTrainConfig, balanced_depth_error, policy_gradient_loss, rewards_to_go
    from traxis.reconstruction import Reconstruction


@unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is required")
class AdaptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bank = analytic_bank()
        cls.scale = SignalScale.fit([cls.bank])

    def reconstruction(self, shape=(128, 128)):
        y, x = np.indices(shape)
        return Reconstruction(np.full(shape, 0.2, dtype=np.float32), ((x + y) / (sum(shape) - 2)).astype(np.float32))

    def test_masked_action_probabilities_and_hidden_nan_isolation(self):
        state = ReplayEnvironment(self.bank).reveal(0)
        config = PolicyConfig(width=4, embedding=16)
        model = MaskedActorCritic(config)
        spatial = spatial_features(state, self.reconstruction(), config)
        logits, value = model(state, self.scale, spatial)
        self.assertEqual(float(Categorical(logits=logits).probs[0].detach()), 0)
        dirty = state.waveforms.copy()
        dirty[~state.observed] = np.nan
        altered = replace(state, waveforms=dirty)
        other, _ = model(altered, self.scale, spatial)
        torch.testing.assert_close(logits, other, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(value))
        self.assertFalse(hasattr(state, "depth_mm"))

    def test_last_available_action_is_finite_and_never_repeats(self):
        env = ReplayEnvironment(self.bank)
        for i in range(27):
            state = env.reveal(i)
        config = PolicyConfig(width=4, embedding=16)
        network = MaskedActorCritic(config)
        features = spatial_features(state, self.reconstruction(), config)
        logits, _ = network(state, self.scale, features)
        dist = Categorical(logits=logits)
        self.assertEqual(int(dist.probs.argmax()), 27)
        self.assertTrue(torch.isfinite(dist.entropy()))
        policy = ActorCriticPolicy(network, self.scale, {})
        self.assertEqual(policy.choose(state, self.reconstruction()), 27)
        state = env.reveal(27)
        with self.assertRaises(ValueError):
            policy.choose(state, self.reconstruction())

    def test_empty_state_uncertainty_requirement_and_checkpoint_roundtrip(self):
        state = ReplayEnvironment(self.bank).state()
        config = PolicyConfig(width=4, embedding=16)
        model = MaskedActorCritic(config)
        policy = ActorCriticPolicy(model, self.scale, {"mode": "debug"})
        self.assertGreaterEqual(policy.choose(state, self.reconstruction()), 0)
        with self.assertRaisesRegex(ValueError, "uncertainty"):
            policy.choose(state, Reconstruction(np.zeros((128, 128))))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            policy.save(path)
            loaded = ActorCriticPolicy.load(path)
            self.assertEqual(loaded.choose(state, self.reconstruction()), policy.choose(state, self.reconstruction()))

    def test_heuristic_changes_with_uncertainty_location_and_masks_selected(self):
        state = ReplayEnvironment(self.bank).state()
        y, x = np.indices((128, 128))
        left = Reconstruction(np.zeros((128, 128)), np.exp(-((x - 10)**2 + (y - 40)**2) / 200))
        right = Reconstruction(np.zeros((128, 128)), np.exp(-((x - 115)**2 + (y - 85)**2) / 200))
        heuristic = UncertaintyPathPolicy()
        a, b = heuristic.choose(state, left), heuristic.choose(state, right)
        self.assertNotEqual(a, b)
        env = ReplayEnvironment(self.bank)
        state = env.reveal(a)
        self.assertNotEqual(heuristic.choose(state, left), a)

    def test_reward_to_go_and_actor_value_gradient_formula(self):
        returns = rewards_to_go([1., 2., 3.], 0.5)
        torch.testing.assert_close(returns, torch.tensor([2.75, 3.5, 3.]))
        logp = torch.tensor(-0.7, requires_grad=True)
        value = torch.tensor(0.2, requires_grad=True)
        entropy = torch.tensor(0.5, requires_grad=True)
        cfg = RLTrainConfig(value_weight=0.4, entropy_weight=0.03)
        loss = policy_gradient_loss(logp, value, entropy, torch.tensor(0.8), 0.9, cfg)
        loss.backward()
        self.assertAlmostEqual(float(logp.grad), -0.9 * 0.6, places=6)
        self.assertAlmostEqual(float(value.grad), 0.4 * -0.6, places=6)
        self.assertAlmostEqual(float(entropy.grad), -0.9 * 0.03, places=6)

    def test_balanced_reward_does_not_hide_a_small_missed_defect(self):
        truth = np.zeros((16, 16))
        truth[0, 0] = 2
        error = balanced_depth_error(np.zeros_like(truth), truth)
        self.assertEqual(error, 1)
        self.assertEqual(balanced_depth_error(np.ones_like(truth), np.zeros_like(truth)), 1)

    def test_action_permutation_equivariance_and_larger_catalogs(self):
        from traxis.acquisition import action_catalog, sensor_layout
        config = PolicyConfig(width=4, embedding=16)
        network = MaskedActorCritic(config).eval()
        state = ReplayEnvironment(self.bank).reveal(0)
        permutation = np.random.default_rng(17).permutation(len(state.actions))
        shuffled = replace(state, actions=tuple(state.actions[i] for i in permutation),
                           waveforms=state.waveforms[permutation], baseline=state.baseline[permutation], observed=state.observed[permutation])
        with torch.inference_mode():
            original, _ = network(state, self.scale, spatial_features(state, self.reconstruction(), config))
            changed, _ = network(shuffled, self.scale, spatial_features(shuffled, self.reconstruction(), config))
            torch.testing.assert_close(changed, original[permutation], rtol=1e-5, atol=1e-5)
        actions = action_catalog(12, (100000., 150000.))
        bank = replace(self.bank, sensors_xy_m=sensor_layout(self.bank.plate, 12), actions=actions,
                       waveforms=np.zeros((len(actions), len(self.bank.time_s))), baseline=np.zeros((len(actions), len(self.bank.time_s))))
        state = ReplayEnvironment(bank).state()
        with torch.inference_mode():
            logits, value = network(state, self.scale, spatial_features(state, self.reconstruction(), config))
        self.assertEqual(len(logits), 132)
        self.assertTrue(torch.isfinite(logits).all())
