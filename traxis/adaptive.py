"""Public-state adaptive sensing models and an explicit path heuristic."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from traxis.bank import Observation
from traxis.models import ModelConfig, RecordEncoder, SignalScale, action_context, observation_batch
from traxis.reconstruction import Reconstruction
from traxis.storage import atomic_write


@dataclass(frozen=True)
class PolicyConfig:
    width: int = 8
    embedding: int = 32
    map_size: int = 16
    path_samples: int = 16
    require_uncertainty: bool = True

    def __post_init__(self):
        ModelConfig(width=self.width, embedding=self.embedding)
        if type(self.map_size) is not int or not 4 <= self.map_size <= 64:
            raise ValueError("policy map size must be an integer in [4,64]")
        if type(self.path_samples) is not int or not 4 <= self.path_samples <= 128:
            raise ValueError("path samples must be an integer in [4,128]")
        if type(self.require_uncertainty) is not bool:
            raise ValueError("require_uncertainty must be Boolean")


@dataclass(frozen=True)
class SpatialFeatures:
    maps: torch.Tensor  # (1, depth/spread/spread-available, map_size, map_size)
    paths: torch.Tensor  # (actions, mean depth, max depth, mean spread, max spread)


def spatial_features(observation: Observation, reconstruction: Reconstruction, config: PolicyConfig) -> SpatialFeatures:
    """Use only inference outputs and public geometry; no rewards or labels."""
    depth = np.asarray(reconstruction.depth_mm, dtype=np.float32)
    spread = reconstruction.std_mm
    if config.require_uncertainty and spread is None:
        raise ValueError("this policy requires an ensemble uncertainty map")
    if depth.ndim != 2 or min(depth.shape) < 1 or not np.isfinite(depth).all():
        raise ValueError("policy depth map must be finite and two-dimensional")
    thickness = observation.plate.thickness_m * 1000
    if np.any(depth < 0) or np.any(depth > thickness + 1e-5):
        raise ValueError("policy depth map lies outside nominal thickness")
    std = np.zeros_like(depth) if spread is None else np.asarray(spread, dtype=np.float32)
    if std.shape != depth.shape or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError("policy spread must be finite, nonnegative and match the depth grid")
    values = torch.from_numpy(np.stack([depth, std]))[None] / thickness
    maps = F.adaptive_avg_pool2d(values, (config.map_size, config.map_size))
    maps = torch.cat([maps, torch.full_like(maps[:, :1], float(spread is not None))], dim=1)
    context = action_context(observation)
    tx, rx = context[:, :2], context[:, 2:4]
    fraction = np.linspace(0, 1, config.path_samples, dtype=np.float32)
    points = tx[:, None, :] + fraction[None, :, None] * (rx - tx)[:, None, :]
    grid = torch.from_numpy(points * 2 - 1)[None]
    sampled = F.grid_sample(values, grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
    paths = torch.stack([sampled[0].mean(-1), sampled[0].amax(-1),
                         sampled[1].mean(-1), sampled[1].amax(-1)], dim=-1)
    return SpatialFeatures(maps, paths)


class MaskedActorCritic(nn.Module):
    """A shared per-action actor and a state-value baseline, independent of M."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        e, w = config.embedding, config.width
        self.records = RecordEncoder(ModelConfig(width=w, embedding=e))
        self.map_encoder = nn.Sequential(nn.Conv2d(3, w, 3, padding=1), nn.GELU(),
                                          nn.Conv2d(w, w * 2, 3, padding=1), nn.GELU(),
                                          nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(w * 8, e))
        self.candidates = nn.Sequential(nn.Linear(16, e), nn.GELU())
        self.actor = nn.Sequential(nn.Linear(e * 3 + 1, e), nn.GELU(), nn.Linear(e, 1))
        self.critic = nn.Sequential(nn.Linear(e * 2 + 1, e), nn.GELU(), nn.Linear(e, 1))

    def forward(self, observation: Observation, scale: SignalScale, spatial: SpatialFeatures):
        if observation.count == len(observation.actions):
            raise ValueError("all records have been acquired")
        expected = (1, 3, self.config.map_size, self.config.map_size)
        if spatial.maps.shape != expected or spatial.paths.shape != (len(observation.actions), 4):
            raise ValueError("policy spatial feature shapes do not match configuration/catalog")
        if not torch.isfinite(spatial.maps).all() or not torch.isfinite(spatial.paths).all():
            raise ValueError("policy spatial features must be finite")
        tokens, padding = self.records(observation_batch([observation], scale, compact=True))
        valid = (~padding).unsqueeze(-1)
        wave = (tokens * valid).sum(dim=1) / valid.sum(dim=1)
        image = self.map_encoder(spatial.maps)
        fraction = torch.tensor([[observation.count / len(observation.actions)]], dtype=wave.dtype)
        state = torch.cat([wave, image, fraction], dim=-1)
        candidate = self.candidates(torch.cat([torch.from_numpy(action_context(observation)), spatial.paths], dim=-1))
        logits = self.actor(torch.cat([state.expand(len(candidate), -1), candidate], dim=-1)).squeeze(-1)
        mask = torch.from_numpy(observation.observed.copy())
        logits = logits.masked_fill(mask, -torch.inf)
        return logits, self.critic(state).squeeze()


class ActorCriticPolicy:
    def __init__(self, network: MaskedActorCritic, scale: SignalScale, provenance: dict):
        self.network, self.scale, self.provenance = network.cpu().eval(), scale, provenance

    def choose(self, observation: Observation, reconstruction: Reconstruction) -> int:
        with torch.inference_mode():
            spatial = spatial_features(observation, reconstruction, self.network.config)
            logits, _ = self.network(observation, self.scale, spatial)
        return int(logits.argmax())

    def save(self, path: Path):
        json.dumps(self.provenance, allow_nan=False)
        state = {"schema_version": 1, "kind": "masked_actor_critic", "config": asdict(self.network.config),
                 "scale": asdict(self.scale), "weights": self.network.state_dict(), "provenance": self.provenance}
        atomic_write(path, lambda stream: torch.save(state, stream))

    @classmethod
    def load(cls, path: Path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["schema_version"] != 1 or state["kind"] != "masked_actor_critic":
            raise ValueError("unsupported adaptive policy artifact")
        with torch.random.fork_rng(devices=[]):
            network = MaskedActorCritic(PolicyConfig(**state["config"]))
        network.load_state_dict(state["weights"], strict=True)
        return cls(network, SignalScale(**state["scale"]), state["provenance"])


class UncertaintyPathPolicy:
    """Mean spread on a straight sensor path; not expected information gain.

    Frequencies on one path tie unless already acquired. This geometric
    heuristic is an additional comparator, not a replacement for trained RL.
    """

    def __init__(self, path_samples: int = 16):
        self.config = PolicyConfig(path_samples=path_samples)

    def choose(self, observation: Observation, reconstruction: Reconstruction) -> int:
        features = spatial_features(observation, reconstruction, self.config)
        scores = features.paths[:, 2].numpy().copy()
        scores[observation.observed] = -np.inf
        if observation.count == len(scores):
            raise ValueError("all records have been acquired")
        return int(scores.argmax())
