"""Partial-observation waveform and DAS-assisted depth reconstruction.

These are configurable research candidates, not pretrained models. Signal
scale must be fitted on training data and kept fixed during evaluation.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from traxis.bank import Observation, WaveformBank, validate_specimen_splits
from traxis.config import positive


@dataclass(frozen=True)
class SignalScale:
    rms: float
    observable: str

    def __post_init__(self):
        positive("signal RMS", self.rms)

    @classmethod
    def fit(cls, banks: list[WaveformBank]) -> "SignalScale":
        if not banks or any(b.metadata.split != "train" for b in banks):
            raise ValueError("signal scaling may only fit nonempty simulation training banks")
        validate_specimen_splits(banks)
        units = {b.metadata.observable for b in banks}
        if len(units) != 1:
            raise ValueError("cannot combine observables with different units")
        square_sum, count = 0., 0
        for bank in banks:
            raw = bank.waveforms.astype(float)
            residual = raw - bank.baseline
            square_sum += float((raw**2).sum() + (residual**2).sum())
            count += raw.size + residual.size
        return cls(math.sqrt(square_sum / count) if square_sum > 0 else 1., next(iter(units)))


@dataclass
class ObservationBatch:
    signals: torch.Tensor  # (batch, records, raw/residual, time)
    context: torch.Tensor  # (batch, records, 12)
    observed: torch.Tensor  # (batch, records), bool


def observation_batch(observations: list[Observation], scale: SignalScale) -> ObservationBatch:
    if not observations:
        raise ValueError("cannot form an empty batch")
    lengths = {len(o.time_s) for o in observations}
    if len(lengths) != 1:
        raise ValueError("batch traces must use a common sample count; resample explicitly upstream")
    max_records = max(len(o.actions) for o in observations)
    signals = np.zeros((len(observations), max_records, 2, next(iter(lengths))), dtype=np.float32)
    context = np.zeros((len(observations), max_records, 12), dtype=np.float32)
    mask = np.zeros((len(observations), max_records), dtype=bool)
    for i, o in enumerate(observations):
        if o.observable != scale.observable:
            raise ValueError("observation and training scale have different physical units")
        n = len(o.actions)
        signals[i, :n, 0] = np.where(o.observed[:, None], o.waveforms, 0) / scale.rms
        signals[i, :n, 1] = np.where(o.observed[:, None], o.waveforms - o.baseline, 0) / scale.rms
        mask[i, :n] = o.observed
        for j, a in enumerate(o.actions):
            tx, rx = o.sensors_xy_m[[a.transmitter, a.receiver]] / [o.plate.width_m, o.plate.height_m]
            context[i, j] = [*tx, *rx, a.transmitter / (len(o.sensors_xy_m) - 1),
                             a.receiver / (len(o.sensors_xy_m) - 1), a.frequency_hz / 1e6,
                             (o.time_s[1] - o.time_s[0]) * 1e6, o.time_s[0] * 1000,
                             o.plate.width_m, o.plate.height_m, o.plate.thickness_m * 1000]
    return ObservationBatch(torch.from_numpy(signals), torch.from_numpy(context), torch.from_numpy(mask))


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "direct"
    output_size: int = 128
    width: int = 16
    embedding: int = 64
    heads: int = 4
    extra_decoder_blocks: int = 0
    max_depth_mm: float = 3.0

    def __post_init__(self):
        if self.architecture not in ("direct", "das_unet"):
            raise ValueError("architecture must be direct or das_unet")
        if self.output_size not in (128, 256):
            raise ValueError("research output sizes are 128 and 256")
        if type(self.width) is not int or not 4 <= self.width <= 64:
            raise ValueError("width must be an integer in [4,64]")
        if type(self.embedding) is not int or not 16 <= self.embedding <= 256:
            raise ValueError("embedding must be an integer in [16,256]")
        if type(self.heads) is not int or self.heads < 1 or self.embedding % self.heads:
            raise ValueError("attention heads must divide embedding size")
        if type(self.extra_decoder_blocks) is not int or not 0 <= self.extra_decoder_blocks <= 3:
            raise ValueError("extra decoder blocks must lie in [0,3]")
        positive("maximum depth", self.max_depth_mm)


class RecordEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        w, e = config.width, config.embedding
        self.temporal = nn.Sequential(nn.Conv1d(2, w, 9, stride=4, padding=4), nn.GroupNorm(1, w), nn.GELU(),
                                      nn.Conv1d(w, w * 2, 7, stride=4, padding=3), nn.GroupNorm(1, w * 2), nn.GELU(),
                                      nn.Conv1d(w * 2, w * 2, 5, stride=2, padding=2), nn.GELU(),
                                      nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(w * 16, e))
        self.geometry = nn.Sequential(nn.Linear(12, e), nn.GELU(), nn.Linear(e, e))
        self.amplitude = nn.Linear(2, e)
        self.norm = nn.LayerNorm(e)
        self.null_record = nn.Parameter(torch.zeros(1, 1, e))

    def forward(self, batch: ObservationBatch) -> tuple[torch.Tensor, torch.Tensor]:
        s, c, mask = batch.signals, batch.context, batch.observed
        if s.ndim != 4 or s.shape[2] != 2 or s.shape[-1] < 64 or s.shape[:2] != mask.shape or c.shape != (*mask.shape, 12):
            raise ValueError("invalid observation tensor shapes or fewer than 64 samples")
        if mask.dtype != torch.bool:
            raise ValueError("observation mask must be Boolean")
        # Mask before convolution/normalization; zero times NaN would still leak.
        s = torch.where(mask[:, :, None, None], s, 0)
        c = torch.where(mask[:, :, None], c, 0)
        if not torch.isfinite(s).all() or not torch.isfinite(c).all():
            raise ValueError("observed signals and context must be finite")
        b, m, _, t = s.shape
        rms = s.square().mean(dim=-1).clamp_min(1e-12).sqrt().log1p()
        encoded = self.norm(self.temporal(s.reshape(b * m, 2, t)).reshape(b, m, -1)
                            + self.geometry(c) + self.amplitude(rms))
        # A null token is available only for empty inspection states. Attention
        # with every key masked otherwise produces NaNs in common backends.
        tokens = torch.cat([encoded, self.null_record.expand(b, -1, -1)], dim=1)
        padding = torch.cat([~mask, mask.any(dim=1, keepdim=True)], dim=1)
        return tokens, padding


class ResidualBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(1, outputs), nn.GELU(),
                                  nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(1, outputs))
        self.skip = nn.Conv2d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x):
        return F.gelu(self.main(x) + self.skip(x))


class DirectReconstructor(nn.Module):
    """Shared 1D CNN -> spatial-query attention -> convolutional map decoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = RecordEncoder(config)
        self.queries = nn.Parameter(torch.randn(1, 64, config.embedding) * 0.02)
        self.attention = nn.MultiheadAttention(config.embedding, config.heads, batch_first=True)
        self.seed = nn.Conv2d(config.embedding, config.width * 4, 1)
        channels = config.width * 4
        blocks = []
        for _ in range(int(math.log2(config.output_size // 8))):
            next_channels = max(config.width, channels // 2)
            blocks.extend([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                           ResidualBlock(channels, next_channels)])
            channels = next_channels
        blocks.extend(ResidualBlock(channels, channels) for _ in range(config.extra_decoder_blocks))
        self.decoder = nn.Sequential(*blocks, nn.Conv2d(channels, 1, 1))

    def forward(self, batch: ObservationBatch) -> torch.Tensor:
        tokens, padding = self.encoder(batch)
        features, _ = self.attention(self.queries.expand(len(tokens), -1, -1), tokens, tokens,
                                     key_padding_mask=padding, need_weights=False)
        seed = features.transpose(1, 2).reshape(len(tokens), self.config.embedding, 8, 8)
        return self.config.max_depth_mm * torch.sigmoid(self.decoder(self.seed(seed)))


class DASUNetReconstructor(nn.Module):
    """Residual U-Net with waveform-attention context fused at its bottleneck.

    Inputs are residual-envelope intensity and timing coverage. The residual
    encoder is trained from scratch; it is not an ImageNet ResNet checkpoint.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        w = config.width
        self.encoder = RecordEncoder(config)
        self.query = nn.Parameter(torch.zeros(1, 1, config.embedding))
        self.attention = nn.MultiheadAttention(config.embedding, config.heads, batch_first=True)
        self.context = nn.Linear(config.embedding, w * 4)
        self.enc1, self.enc2, self.bottom = ResidualBlock(2, w), ResidualBlock(w, w * 2), ResidualBlock(w * 2, w * 4)
        self.dec2, self.dec1 = ResidualBlock(w * 6, w * 2), ResidualBlock(w * 3, w)
        self.refine = nn.Sequential(*(ResidualBlock(w, w) for _ in range(config.extra_decoder_blocks)))
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, batch: ObservationBatch, das: torch.Tensor) -> torch.Tensor:
        size = self.config.output_size
        if das.shape != (batch.signals.shape[0], 2, size, size) or not torch.isfinite(das).all():
            raise ValueError("DAS input must be finite (batch, intensity/coverage, output_size, output_size)")
        tokens, padding = self.encoder(batch)
        fused, _ = self.attention(self.query.expand(len(tokens), -1, -1), tokens, tokens,
                                  key_padding_mask=padding, need_weights=False)
        a = self.enc1(das)
        b = self.enc2(F.avg_pool2d(a, 2))
        z = self.bottom(F.avg_pool2d(b, 2)) + self.context(fused[:, 0])[:, :, None, None]
        z = self.dec2(torch.cat([F.interpolate(z, size=b.shape[-2:], mode="bilinear", align_corners=False), b], dim=1))
        z = self.dec1(torch.cat([F.interpolate(z, size=a.shape[-2:], mode="bilinear", align_corners=False), a], dim=1))
        return self.config.max_depth_mm * torch.sigmoid(self.head(self.refine(z)))


def build_model(config: ModelConfig) -> nn.Module:
    return DirectReconstructor(config) if config.architecture == "direct" else DASUNetReconstructor(config)


def reconstruction_loss(prediction_mm: torch.Tensor, truth_mm: torch.Tensor,
                         damaged_weight: float = 10.) -> torch.Tensor:
    """Balance defect and intact regions instead of hiding defects in plate area."""
    positive("damaged region weight", damaged_weight)
    if prediction_mm.shape != truth_mm.shape or truth_mm.ndim != 4 or truth_mm.shape[1] != 1:
        raise ValueError("prediction and truth must have matching (batch,1,y,x) grids")
    if not torch.isfinite(truth_mm).all() or (truth_mm < 0).any():
        raise ValueError("depth labels must be finite and nonnegative")
    weights = torch.where(truth_mm > 0, damaged_weight, 1.)
    return (weights * (prediction_mm - truth_mm).square()).sum() / weights.sum()


def save_model(path: Path, model: nn.Module, scale: SignalScale, provenance: dict) -> None:
    """Store tensors and plain metadata only; never pickle a model object."""
    if not isinstance(model, (DirectReconstructor, DASUNetReconstructor)):
        raise ValueError("unsupported reconstruction model")
    json.dumps(provenance, allow_nan=False)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "config": asdict(model.config), "scale": asdict(scale),
                "state_dict": model.state_dict(), "provenance": provenance}, path)


def load_model(path: Path) -> tuple[nn.Module, SignalScale, dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["schema_version"] != 1:
        raise ValueError("unsupported model checkpoint schema")
    model = build_model(ModelConfig(**state["config"]))
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    return model, SignalScale(**state["scale"]), state["provenance"]
