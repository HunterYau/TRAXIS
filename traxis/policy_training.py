"""On-policy reward-to-go policy gradients with a learned value baseline.

The estimator follows https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html .
It uses full finite-inspection returns, not PPO or bootstrapped A2C. The
reconstructor is frozen; labels are used only for training/validation rewards.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from traxis.adaptive import ActorCriticPolicy, MaskedActorCritic, PolicyConfig, spatial_features
from traxis.bank import ReplayEnvironment
from traxis.config import positive
from traxis.dataset import BankDataset
from traxis.metrics import resample_cell_map
from traxis.models import SignalScale
from traxis.reconstruction import EnsembleReconstructor, NeuralReconstructor, load_reconstruction_bundle
from traxis.resources import ResourceGuard
from traxis.storage import atomic_write, canonical_hash, sha256_file
from traxis.training import fit_signal_scale


@dataclass(frozen=True)
class RLTrainConfig:
    mode: str = "debug"
    max_episodes: int = 2
    horizon: int = 4
    initial_actions: tuple[int, ...] = ()
    seed: int = 2026
    learning_rate: float = 3e-4
    gamma: float = 0.95
    record_cost: float = 0.001
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    gradient_clip: float = 1.
    validation_interval: int = 1
    patience: int = 10
    cpu_threads: int = 1

    def __post_init__(self):
        if self.mode not in ("debug", "research"):
            raise ValueError("policy training mode must be debug or research")
        for name in ("max_episodes", "horizon", "validation_interval", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.mode == "debug" and (self.max_episodes > 4 or self.horizon > 8):
            raise ValueError("debug RL is limited to four episodes of at most eight acquisitions")
        if type(self.cpu_threads) is not int or not 1 <= self.cpu_threads <= 8:
            raise ValueError("policy CPU threads must be an integer in [1,8]")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not math.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must lie in (0,1]")
        for name in ("learning_rate", "gradient_clip"):
            positive(name, getattr(self, name))
        for name in ("record_cost", "value_weight", "entropy_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        initial = tuple(self.initial_actions)
        if any(type(i) is not int or i < 0 for i in initial) or len(set(initial)) != len(initial):
            raise ValueError("initial actions must be unique nonnegative indices")
        object.__setattr__(self, "initial_actions", initial)


def balanced_depth_error(prediction, truth) -> float:
    """Equal weighting of damaged/intact regions; intact controls use plate MAE."""
    truth = np.asarray(truth, dtype=float)
    prediction = resample_cell_map(prediction, truth.shape)
    if not np.isfinite(truth).all() or np.any(truth < 0) or np.any(prediction < 0):
        raise ValueError("reward maps must be finite and nonnegative")
    error, damaged = abs(prediction - truth), truth > 0
    regions = [error[mask].mean() for mask in (damaged, ~damaged) if mask.any()]
    return float(np.mean(regions))


def rewards_to_go(rewards: list[float], gamma: float) -> torch.Tensor:
    if not rewards or not all(math.isfinite(r) for r in rewards) or not 0 < gamma <= 1:
        raise ValueError("returns need finite rewards and gamma in (0,1]")
    result, running = [], 0.
    for reward in reversed(rewards):
        running = reward + gamma * running
        result.append(running)
    return torch.tensor(list(reversed(result)), dtype=torch.float32)


def policy_gradient_loss(log_probability, value, entropy, target, discount: float, config: RLTrainConfig):
    """Detach the return-minus-value baseline only in the actor term."""
    advantage = (target - value).detach()
    actor = -discount * log_probability * advantage
    critic = 0.5 * config.value_weight * (value - target).square()
    return actor + critic - discount * config.entropy_weight * entropy


def _source_hash():
    root = Path(__file__).parent
    return canonical_hash({name: sha256_file(root / name) for name in
                           ("policy_training.py", "adaptive.py", "models.py", "bank.py", "reconstruction.py", "metrics.py")})


class PolicyTrainingSession:
    """CPU policy learning with fixed-budget episodes and bounded graph memory.

    Collect actions without autograd, then replay public states to accumulate
    policy gradients one state at a time. Weights stay unchanged throughout
    collection/replay, preserving the on-policy estimator without retaining
    a full inspection's convolution graphs or waveform tensors.
    """

    def __init__(self, dataset: BankDataset, reconstruction_paths: list[Path],
                  policy_config: PolicyConfig, config: RLTrainConfig, guard: ResourceGuard | None = None):
        self.dataset, self.policy_config, self.config, self.guard = dataset, policy_config, config, guard
        torch.set_num_threads(config.cpu_threads)
        self.fit_entries, self.validation_entries = dataset.select("train"), dataset.select("validation")
        if not self.fit_entries:
            raise ValueError("RL fitting requires simulation training specimens")
        self.predictor, self.reconstruction_provenances, self.artifacts = load_reconstruction_bundle(reconstruction_paths, guard)
        relevant = self.fit_entries + self.validation_entries
        if config.mode == "research" and (any(e.validation_status == "plumbing_only" or e.observable == "arbitrary" for e in relevant)
                                          or any(p.get("mode") != "research" for p in self.reconstruction_provenances)):
            raise ValueError("research policy training requires verified data and research reconstruction artifacts")
        if policy_config.require_uncertainty and not isinstance(self.predictor, EnsembleReconstructor):
            raise ValueError("uncertainty-aware RL requires an independently seeded reconstruction ensemble")
        inherited_fit = set().union(*(set(p["fit_parents"]) for p in self.reconstruction_provenances))
        inherited_validation = set().union(*(set(p["validation_parents"]) for p in self.reconstruction_provenances))
        final_parents = {e.parent_specimen_id for e in dataset.entries if e.split in ("test", "physical_holdout")}
        if final_parents & (inherited_fit | inherited_validation):
            raise ValueError("reconstructor was exposed to a final policy-evaluation parent")
        if inherited_fit & {e.parent_specimen_id for e in self.validation_entries}:
            raise ValueError("RL validation parent was already used to fit the reconstructor")
        self.inherited_fit, self.inherited_validation = inherited_fit, inherited_validation
        if isinstance(self.predictor, EnsembleReconstructor):
            self.scale = self.predictor.members[0].scale
        elif isinstance(self.predictor, NeuralReconstructor):
            self.scale = self.predictor.scale
        else:
            self.scale = fit_signal_scale(dataset, guard)
        if any(e.observable != self.scale.observable for e in relevant):
            raise ValueError("policy and reconstruction data must share signal units")
        torch.manual_seed(config.seed)
        self.network = MaskedActorCritic(policy_config)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.specimen_rng = np.random.default_rng(config.seed)
        self.action_rng = torch.Generator().manual_seed(config.seed + 1)
        self.episode_count, self.history, self.validation_history = 0, [], []
        self.best_score, self.best_episode, self.bad_validations, self.best_weights = None, None, 0, None
        self.pending_validation = False
        self._owned_checkpoints = set()

    def _check(self, force=False):
        if self.guard:
            self.guard.check(force=force)

    def _environment(self, bank):
        env = ReplayEnvironment(bank)
        for index in self.config.initial_actions:
            env.reveal(index)
        if len(bank.actions) - env.state().count < self.config.horizon:
            raise ValueError("episode horizon exceeds remaining eligible actions")
        return env

    def train_episode(self):
        if self.episode_count >= self.config.max_episodes:
            raise ValueError("configured policy episode budget is exhausted")
        self._check(force=True)
        entry = self.fit_entries[int(self.specimen_rng.integers(len(self.fit_entries)))]
        bank = self.dataset.read(entry)
        env, trajectory = self._environment(bank), []
        self.network.train()
        reconstruction = self.predictor.predict(env.state())
        previous = balanced_depth_error(reconstruction.depth_mm, bank.depth_mm)
        initial_error = previous
        for _ in range(self.config.horizon):
            self._check()
            observation = env.state()
            spatial = spatial_features(observation, reconstruction, self.policy_config)
            with torch.no_grad():
                logits, _ = self.network(observation, self.scale, spatial)
                distribution = Categorical(logits=logits)
                action = int(torch.multinomial(distribution.probs, 1, generator=self.action_rng))
            reconstruction = self.predictor.predict(env.reveal(action))
            error = balanced_depth_error(reconstruction.depth_mm, bank.depth_mm)
            reward = (previous - error) / (bank.plate.thickness_m * 1000) - self.config.record_cost
            trajectory.append((action, float(reward), spatial))
            previous = error
        returns = rewards_to_go([row[1] for row in trajectory], self.config.gamma)
        replay = self._environment(bank)
        self.optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        for t, (action, _reward, spatial) in enumerate(trajectory):
            self._check()
            logits, value = self.network(replay.state(), self.scale, spatial)
            distribution = Categorical(logits=logits)
            loss = policy_gradient_loss(distribution.log_prob(torch.tensor(action)), value,
                                        distribution.entropy(), returns[t], self.config.gamma**t, self.config) / len(trajectory)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite actor/value objective")
            loss.backward()
            loss_value += float(loss.detach())
            replay.reveal(action)
        norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
        self._check(force=True)
        self.optimizer.step()
        self.episode_count += 1
        row = {"episode": self.episode_count, "specimen_id": entry.specimen_id,
               "action_indices": [r[0] for r in trajectory], "rewards": [r[1] for r in trajectory],
               "returns": returns.tolist(), "loss": loss_value, "gradient_norm_before_clip": float(norm),
               "initial_balanced_error_mm": initial_error, "final_balanced_error_mm": previous}
        self.history.append(row)
        self.pending_validation = self.episode_count % self.config.validation_interval == 0
        return row

    def validate(self):
        if not self.validation_entries:
            self.pending_validation = False
            return None
        scores = []
        policy = ActorCriticPolicy(self.network, self.scale, {})
        for entry in self.validation_entries:
            bank = self.dataset.read(entry)
            env = self._environment(bank)
            errors = []
            reconstruction = self.predictor.predict(env.state())
            for _ in range(self.config.horizon):
                self._check()
                action = policy.choose(env.state(), reconstruction)
                reconstruction = self.predictor.predict(env.reveal(action))
                errors.append(balanced_depth_error(reconstruction.depth_mm, bank.depth_mm))
            scores.append(float(np.mean(errors)))
        score = float(np.mean(scores))
        self.validation_history.append({"episode": self.episode_count, "mean_prefix_balanced_error_mm": score})
        if self.best_score is None or score < self.best_score:
            self.best_score, self.best_episode, self.bad_validations = score, self.episode_count, 0
            self.best_weights = {k: v.detach().clone() for k, v in self.network.state_dict().items()}
        else:
            self.bad_validations += 1
        self.pending_validation = False
        return score

    def provenance(self):
        return {"mode": self.config.mode, "algorithm": "finite-horizon reward-to-go with learned value baseline",
                "seed": self.config.seed, "episodes": self.episode_count, "dataset_fingerprint": self.dataset.fingerprint,
                "fit_parents": sorted({e.parent_specimen_id for e in self.fit_entries} | self.inherited_fit),
                "validation_parents": sorted({e.parent_specimen_id for e in self.validation_entries} | self.inherited_validation),
                "policy_fit_parents": sorted({e.parent_specimen_id for e in self.fit_entries}),
                "reconstruction_artifacts": self.artifacts, "training": asdict(self.config),
                "source_hash": _source_hash(), "best_validation_episode": self.best_episode}

    def run(self, episodes: int, checkpoint: Path | None = None):
        if type(episodes) is not int or episodes < 1 or self.episode_count + episodes > self.config.max_episodes:
            raise ValueError("requested episodes exceed the configured total budget")
        if checkpoint and Path(checkpoint).exists() and Path(checkpoint).resolve() not in self._owned_checkpoints:
            raise FileExistsError("resume the existing policy checkpoint explicitly")
        if self.pending_validation:
            self.validate()
        rows = []
        for _ in range(episodes):
            if self.bad_validations >= self.config.patience:
                break
            rows.append(self.train_episode())
            if checkpoint:
                self.save_checkpoint(checkpoint, overwrite=Path(checkpoint).exists())
                self._owned_checkpoints.add(Path(checkpoint).resolve())
            if self.pending_validation:
                self.validate()
                if checkpoint:
                    self.save_checkpoint(checkpoint, overwrite=True)
            self._check(force=True)
        return rows

    def save_checkpoint(self, path: Path, *, overwrite=False):
        state = {"schema_version": 1, "policy_config": asdict(self.policy_config), "training": asdict(self.config),
                 "weights": self.network.state_dict(), "optimizer": self.optimizer.state_dict(), "scale": asdict(self.scale),
                 "torch_rng": torch.get_rng_state(), "specimen_rng": self.specimen_rng.bit_generator.state,
                 "action_rng": self.action_rng.get_state(), "episode_count": self.episode_count, "history": self.history,
                 "validation_history": self.validation_history, "best_score": self.best_score, "best_episode": self.best_episode,
                 "best_weights": self.best_weights, "bad_validations": self.bad_validations,
                 "pending_validation": self.pending_validation, "provenance": self.provenance()}
        atomic_write(path, lambda stream: torch.save(state, stream), overwrite=overwrite)

    @classmethod
    def resume(cls, path: Path, dataset: BankDataset, guard=None):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["schema_version"] != 1 or state["provenance"]["source_hash"] != _source_hash() or state["provenance"]["dataset_fingerprint"] != dataset.fingerprint:
            raise ValueError("policy checkpoint source/dataset mismatch")
        artifacts = state["provenance"]["reconstruction_artifacts"]
        if any(sha256_file(Path(a["path"])) != a["sha256"] for a in artifacts):
            raise ValueError("bound reconstruction artifact changed")
        session = cls(dataset, [Path(a["path"]) for a in artifacts], PolicyConfig(**state["policy_config"]),
                      RLTrainConfig(**state["training"]), guard)
        session.network.load_state_dict(state["weights"], strict=True)
        session.optimizer.load_state_dict(state["optimizer"])
        session.scale = SignalScale(**state["scale"])
        for name in ("episode_count", "history", "validation_history", "best_score", "best_episode", "best_weights", "bad_validations", "pending_validation"):
            setattr(session, name, state[name])
        torch.set_rng_state(state["torch_rng"])
        session.specimen_rng.bit_generator.state = state["specimen_rng"]
        session.action_rng.set_state(state["action_rng"])
        session._owned_checkpoints.add(Path(path).resolve())
        return session

    def export_policy(self, path: Path, *, best_validation=True):
        with torch.random.fork_rng(devices=[]):
            network = MaskedActorCritic(self.policy_config)
        if best_validation and self.validation_entries:
            if self.best_weights is None:
                raise ValueError("validate the selector before exporting selected weights")
            network.load_state_dict(self.best_weights)
        else:
            network.load_state_dict(self.network.state_dict())
        provenance = {**self.provenance(), "weights_episode": self.best_episode if best_validation and self.validation_entries else self.episode_count}
        ActorCriticPolicy(network, self.scale, provenance).save(path)
