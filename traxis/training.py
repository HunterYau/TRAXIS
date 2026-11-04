"""Bounded subset training, deterministic validation and resumable state."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch

from traxis.bank import ReplayEnvironment
from traxis.config import positive
from traxis.dataset import BankDataset
from traxis.imaging import DASSettings
from traxis.metrics import resample_cell_map
from traxis.models import ModelConfig, SignalScale, build_model, load_model, reconstruction_loss, save_model
from traxis.reconstruction import model_inputs
from traxis.resources import ResourceGuard
from traxis.storage import atomic_write, canonical_hash, sha256_file


@dataclass(frozen=True)
class TrainConfig:
    mode: str = "debug"
    phase: str = "pretrain"
    max_steps: int = 2
    seed: int = 2026
    subset_seed: int = 37
    learning_rate: float = 1e-3
    min_records: int = 0
    max_records: int | None = None
    damaged_weight: float = 10.
    gradient_clip: float = 1.
    validation_interval: int = 1
    validation_records: int = 4
    patience: int = 10
    device: str = "cpu"

    def __post_init__(self):
        if self.mode not in ("debug", "research") or self.phase not in ("pretrain", "adaptation"):
            raise ValueError("unknown training mode/phase")
        if type(self.max_steps) is not int or self.max_steps < 1 or (self.mode == "debug" and self.max_steps > 8):
            raise ValueError("debug runs are limited to eight total updates; research steps must be explicit")
        for name in ("seed", "subset_seed", "min_records", "validation_records"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("validation_interval", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_records is not None and (type(self.max_records) is not int or self.max_records < self.min_records):
            raise ValueError("max records must be at least min records")
        for name in ("learning_rate", "damaged_weight", "gradient_clip"):
            positive(name, getattr(self, name))
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")


def _source_hash() -> str:
    root = Path(__file__).parent
    return canonical_hash({name: sha256_file(root / name) for name in
                           ("training.py", "models.py", "reconstruction.py", "imaging.py", "dataset.py", "bank.py")})


def fit_signal_scale(dataset: BankDataset, guard: ResourceGuard | None = None) -> SignalScale:
    entries = dataset.select("train")
    if not entries or len({e.observable for e in entries}) != 1:
        raise ValueError("need simulation training data with one observable unit")
    total, count = 0., 0
    for entry in entries:
        if guard:
            guard.check()
        bank = dataset.read(entry)
        raw = bank.waveforms.astype(float)
        residual = raw - bank.baseline
        total += float((raw**2).sum() + (residual**2).sum())
        count += raw.size + residual.size
    return SignalScale(math.sqrt(total / count) if total else 1., entries[0].observable)


def subset_state(bank, rng: np.random.Generator, count: int):
    if not 0 <= count <= len(bank.actions):
        raise ValueError("subset size exceeds available records")
    env = ReplayEnvironment(bank)
    for index in rng.choice(len(bank.actions), count, replace=False):
        env.reveal(int(index))
    return env.state()


class TrainingSession:
    """One specimen/subset per update; the target never enters model_inputs.

    No implicit epoch loop or GPU choice. Debug mode accepts labeled plumbing
    fixtures for verification only; research mode rejects plumbing-only data.
    Adaptation requires a supplied simulation model and frozen signal scale.
    """

    def __init__(self, dataset: BankDataset, model_config: ModelConfig, config: TrainConfig,
                  das_settings: DASSettings | None = None, *, guard: ResourceGuard | None = None,
                  initial_model=None, frozen_scale: SignalScale | None = None,
                  prior_lineage: dict | None = None, _resume_scale: SignalScale | None = None):
        self.dataset, self.config, self.model_config = dataset, config, model_config
        self.das_settings, self.guard = das_settings, guard
        self.fit_split = "train" if config.phase == "pretrain" else "adaptation"
        self.fit_entries = dataset.select(self.fit_split)
        self.validation_entries = dataset.select("validation") if config.phase == "pretrain" else ()
        if not self.fit_entries:
            raise ValueError(f"dataset has no {self.fit_split} specimens")
        if model_config.architecture == "das_unet" and das_settings is None:
            raise ValueError("DAS-assisted training requires imaging settings")
        relevant = self.fit_entries + self.validation_entries
        if len({e.observable for e in relevant}) != 1:
            raise ValueError("training and validation observables must use matching units")
        if config.mode == "research" and any(e.validation_status == "plumbing_only" or e.observable == "arbitrary" for e in relevant):
            raise ValueError("research training requires verified physical-unit data, not plumbing fixtures")
        if config.phase == "pretrain" and (initial_model is not None or frozen_scale is not None):
            raise ValueError("use checkpoint resume for pretraining recovery")
        if config.phase == "adaptation" and (initial_model is None or frozen_scale is None or prior_lineage is None):
            raise ValueError("physical adaptation requires an existing model and frozen simulation scale")
        if config.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; choose cpu or install a suitable GPU PyTorch build")
        torch.manual_seed(config.seed)
        if initial_model is not None and initial_model.config != model_config:
            raise ValueError("initial model configuration mismatch")
        self.model = build_model(model_config).to(config.device)
        if initial_model is not None:
            self.model.load_state_dict(initial_model.state_dict())
        self.scale = frozen_scale or _resume_scale or fit_signal_scale(dataset, guard)
        if self.scale.observable != relevant[0].observable:
            raise ValueError("adaptation scale and measurements have different physical units")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.rng = np.random.default_rng(config.subset_seed)
        self.step_count, self.history, self.validation_history = 0, [], []
        self.best_loss, self.best_step, self.bad_validations = None, None, 0
        self.best_weights = None
        self.pending_validation = False
        self.parent_checkpoint_sha256 = None
        self.prior_lineage = prior_lineage or {"fit_parents": [], "validation_parents": []}
        self._owned_checkpoints = set()

    def _check(self, force=False):
        if self.guard:
            self.guard.check(force=force)

    def train_step(self) -> dict:
        if self.step_count >= self.config.max_steps:
            raise ValueError("configured total update budget is exhausted")
        self._check(force=True)
        entry = self.fit_entries[int(self.rng.integers(len(self.fit_entries)))]
        bank = self.dataset.read(entry)
        upper = len(bank.actions) if self.config.max_records is None else min(self.config.max_records, len(bank.actions))
        if upper < self.config.min_records:
            raise ValueError("specimen has fewer records than the configured minimum")
        count = int(self.rng.integers(self.config.min_records, upper + 1))
        observation = subset_state(bank, self.rng, count)
        self.model.train()
        args = model_inputs(observation, self.model, self.scale, self.das_settings, self.config.device, self.guard)
        truth = torch.from_numpy(resample_cell_map(bank.depth_mm, (self.model_config.output_size,) * 2).astype(np.float32))[None, None].to(self.config.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss = reconstruction_loss(self.model(*args), truth, self.config.damaged_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
        self._check(force=True)
        self.optimizer.step()
        self.step_count += 1
        row = {"step": self.step_count, "specimen_id": entry.specimen_id, "records": count,
               "action_indices": np.flatnonzero(observation.observed).tolist(),
               "loss_mm2": float(loss.detach()), "gradient_norm_before_clip": float(norm)}
        self.history.append(row)
        self.pending_validation = self.step_count % self.config.validation_interval == 0
        return row

    def validate(self) -> float | None:
        """Fixed validation subsets; physical holdouts are never model selection data."""
        if not self.validation_entries:
            self.pending_validation = False
            return None
        self.model.eval()
        losses = []
        with torch.inference_mode():
            for entry in self.validation_entries:
                self._check()
                bank = self.dataset.read(entry)
                seed = int(canonical_hash([self.config.subset_seed, entry.specimen_id, "validation"])[:16], 16)
                observation = subset_state(bank, np.random.default_rng(seed), min(self.config.validation_records, len(bank.actions)))
                truth = torch.from_numpy(resample_cell_map(bank.depth_mm, (self.model_config.output_size,) * 2).astype(np.float32))[None, None].to(self.config.device)
                loss = reconstruction_loss(self.model(*model_inputs(observation, self.model, self.scale,
                                            self.das_settings, self.config.device, self.guard)), truth, self.config.damaged_weight)
                losses.append(float(loss))
        score = float(np.mean(losses))
        if not math.isfinite(score):
            raise FloatingPointError("nonfinite validation loss")
        self.validation_history.append({"step": self.step_count, "loss_mm2": score})
        if self.best_loss is None or score < self.best_loss:
            self.best_loss, self.best_step, self.bad_validations = score, self.step_count, 0
            self.best_weights = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        else:
            self.bad_validations += 1
        self.pending_validation = False
        return score

    def run(self, additional_steps: int, checkpoint: Path | None = None) -> list[dict]:
        if type(additional_steps) is not int or additional_steps < 1 or self.step_count + additional_steps > self.config.max_steps:
            raise ValueError("requested updates exceed the configured total budget")
        if checkpoint and Path(checkpoint).exists() and Path(checkpoint).resolve() not in self._owned_checkpoints:
            raise FileExistsError("refusing to replace a checkpoint from another session; resume it explicitly")
        if self.pending_validation:
            self.validate()
        rows = []
        for _ in range(additional_steps):
            if self.bad_validations >= self.config.patience:
                break
            rows.append(self.train_step())
            # Publish the completed update before possibly expensive validation.
            if checkpoint:
                self.save_checkpoint(checkpoint, overwrite=Path(checkpoint).exists())
                self._owned_checkpoints.add(Path(checkpoint).resolve())
            if self.pending_validation:
                self.validate()
                if checkpoint:
                    self.save_checkpoint(checkpoint, overwrite=True)
            self._check(force=True)
        return rows

    def provenance(self) -> dict:
        return {"mode": self.config.mode, "phase": self.config.phase, "seed": self.config.seed,
                "subset_seed": self.config.subset_seed, "dataset_fingerprint": self.dataset.fingerprint,
                "fit_parents": sorted({e.parent_specimen_id for e in self.fit_entries} | set(self.prior_lineage["fit_parents"])),
                "validation_parents": sorted({e.parent_specimen_id for e in self.validation_entries} | set(self.prior_lineage["validation_parents"])),
                "inherited_lineage": self.prior_lineage,
                "steps": self.step_count, "best_validation_step": self.best_step,
                "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
                "source_hash": _source_hash(), "torch_version": str(torch.__version__),
                "das_settings": asdict(self.das_settings) if self.das_settings else None}

    def save_checkpoint(self, path: Path, *, overwrite: bool = False):
        state = {"schema_version": 1, "model_config": asdict(self.model_config), "train_config": asdict(self.config),
                 "das_settings": asdict(self.das_settings) if self.das_settings else None,
                 "scale": asdict(self.scale), "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if self.config.device == "cuda" else [],
                 "subset_rng": self.rng.bit_generator.state, "step_count": self.step_count,
                 "history": self.history, "validation_history": self.validation_history,
                 "best_loss": self.best_loss, "best_step": self.best_step, "bad_validations": self.bad_validations,
                 "best_weights": self.best_weights, "provenance": self.provenance()}
        state["pending_validation"] = self.pending_validation
        atomic_write(path, lambda stream: torch.save(state, stream), overwrite=overwrite)

    @classmethod
    def resume(cls, path: Path, dataset: BankDataset, guard: ResourceGuard | None = None):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["schema_version"] != 1 or state["provenance"]["dataset_fingerprint"] != dataset.fingerprint:
            raise ValueError("training checkpoint/dataset mismatch")
        if state["provenance"]["source_hash"] != _source_hash():
            raise ValueError("training source changed; use the recorded revision for exact resume")
        config, model_config = TrainConfig(**state["train_config"]), ModelConfig(**state["model_config"])
        das = DASSettings.from_dict(state["das_settings"]) if state["das_settings"] else None
        scale = SignalScale(**state["scale"])
        kwargs = {"initial_model": build_model(model_config), "frozen_scale": scale,
                  "prior_lineage": state["provenance"]["inherited_lineage"]} if config.phase == "adaptation" else {"_resume_scale": scale}
        result = cls(dataset, model_config, config, das, guard=guard, **kwargs)
        result.scale = scale
        result.model.load_state_dict(state["model"], strict=True)
        result.optimizer.load_state_dict(state["optimizer"])
        for name in ("step_count", "history", "validation_history", "best_loss", "best_step", "bad_validations", "best_weights", "pending_validation"):
            setattr(result, name, state[name])
        result.parent_checkpoint_sha256 = state["provenance"]["parent_checkpoint_sha256"]
        result.rng.bit_generator.state = state["subset_rng"]
        torch.set_rng_state(state["torch_rng"])
        if config.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        result._owned_checkpoints.add(Path(path).resolve())
        return result

    @classmethod
    def adapt(cls, model_path: Path, dataset: BankDataset, config: TrainConfig,
               das_settings: DASSettings | None = None, guard: ResourceGuard | None = None):
        if config.phase != "adaptation":
            raise ValueError("adapt requires the adaptation training phase")
        model, scale, provenance = load_model(model_path)
        if not {"fit_parents", "validation_parents"} <= provenance.keys():
            raise ValueError("initial model lacks training/validation parent provenance")
        lineage = {key: provenance[key] for key in ("fit_parents", "validation_parents")}
        final_parents = {e.parent_specimen_id for e in dataset.entries if e.split in ("physical_holdout", "uncertainty_calibration")}
        if final_parents & (set(lineage["fit_parents"]) | set(lineage["validation_parents"])):
            raise ValueError("a final holdout/calibration parent was already used by the initial model")
        session = cls(dataset, model.config, config, das_settings, guard=guard, initial_model=model,
                      frozen_scale=scale, prior_lineage=lineage)
        session.parent_checkpoint_sha256 = sha256_file(model_path)
        return session

    def export_model(self, path: Path, *, best_validation: bool = True):
        with torch.random.fork_rng(devices=[]):
            model = build_model(self.model_config)
        if best_validation and self.validation_entries:
            if self.best_weights is None:
                raise ValueError("validate the model before exporting a selected checkpoint")
            model.load_state_dict(self.best_weights)
        else:
            model.load_state_dict(self.model.state_dict())
        provenance = {**self.provenance(), "weights_step": self.best_step if best_validation and self.validation_entries else self.step_count}
        save_model(path, model, self.scale, provenance)
