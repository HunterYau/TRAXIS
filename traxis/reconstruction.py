"""Adapters from public observations to quantitative reconstruction outputs."""

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
import torch

from traxis.bank import Observation
from traxis.imaging import DASSettings, delay_and_sum
from traxis.metrics import ensemble_statistics
from traxis.models import SignalScale, observation_batch
from traxis.resources import ResourceGuard


@dataclass(frozen=True)
class Reconstruction:
    depth_mm: np.ndarray
    std_mm: np.ndarray | None = None


class Reconstructor(Protocol):
    def predict(self, observation: Observation) -> Reconstruction: ...


def model_inputs(observation: Observation, model, scale: SignalScale, das_settings: DASSettings | None,
                  device: str = "cpu", guard: ResourceGuard | None = None):
    if not np.isclose(observation.plate.thickness_m * 1000, model.config.max_depth_mm):
        raise ValueError("model output bound must match nominal specimen thickness")
    batch = observation_batch([observation], scale, compact=True)
    batch.signals, batch.context, batch.observed = [t.to(device) for t in (batch.signals, batch.context, batch.observed)]
    if model.config.architecture == "direct":
        return (batch,)
    if das_settings is None:
        raise ValueError("DAS-assisted model requires explicit imaging settings")
    size = model.config.output_size
    image = delay_and_sum(observation, replace(das_settings, shape=(size, size)), guard)
    das = torch.from_numpy(np.stack([image.intensity / scale.rms, image.coverage]))[None].to(device)
    return batch, das


class NeuralReconstructor:
    def __init__(self, model, scale: SignalScale, das_settings: DASSettings | None = None,
                  device: str = "cpu", guard: ResourceGuard | None = None):
        self.model, self.scale, self.das_settings = model.to(device).eval(), scale, das_settings
        self.device, self.guard = device, guard

    def predict(self, observation: Observation) -> Reconstruction:
        if self.guard:
            self.guard.check()
        with torch.inference_mode():
            output = self.model(*model_inputs(observation, self.model, self.scale, self.das_settings, self.device, self.guard))
        values = output[0, 0].cpu().numpy().copy()
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite reconstruction")
        if self.guard:
            self.guard.check()
        return Reconstruction(values)


class EnsembleReconstructor:
    def __init__(self, members: list[NeuralReconstructor]):
        if len(members) < 2 or len({id(m.model) for m in members}) != len(members):
            raise ValueError("ensemble requires distinct model instances")
        first = members[0]
        if any(m.model.config != first.model.config or m.scale != first.scale or m.das_settings != first.das_settings for m in members):
            raise ValueError("ensemble members must share configuration, units, scaling and imaging settings")
        self.members = members

    def predict(self, observation: Observation) -> Reconstruction:
        maps = np.stack([m.predict(observation).depth_mm for m in self.members])
        mean, spread = ensemble_statistics(maps)
        return Reconstruction(mean, spread)
