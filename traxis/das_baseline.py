"""DAS-only localization with an explicitly fitted scalar depth conversion."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from traxis.bank import Observation
from traxis.dataset import BankDataset
from traxis.imaging import DASSettings, delay_and_sum
from traxis.metrics import resample_cell_map
from traxis.reconstruction import Reconstruction
from traxis.resources import ResourceGuard
from traxis.storage import write_json
from traxis.training import subset_state


@dataclass(frozen=True)
class DASDepthCalibration:
    slope_mm_per_signal_unit: float
    observable: str
    settings: DASSettings
    provenance: dict

    def __post_init__(self):
        if not math.isfinite(self.slope_mm_per_signal_unit) or self.slope_mm_per_signal_unit <= 0:
            raise ValueError("DAS depth slope must be finite and positive")

    def predict(self, observation: Observation) -> Reconstruction:
        if observation.observable != self.observable:
            raise ValueError("DAS calibration and observation signal units differ")
        image = delay_and_sum(observation, self.settings)
        depth = np.clip(image.intensity * self.slope_mm_per_signal_unit, 0, observation.plate.thickness_m * 1000)
        return Reconstruction(depth)

    def save(self, path: Path):
        write_json(path, {"schema_version": 1, "kind": "das_depth_calibration",
                          "slope_mm_per_signal_unit": self.slope_mm_per_signal_unit,
                          "observable": self.observable, "settings": asdict(self.settings),
                          "provenance": self.provenance})

    @classmethod
    def load(cls, path: Path):
        spec = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if spec["schema_version"] != 1 or spec["kind"] != "das_depth_calibration":
            raise ValueError("unsupported DAS baseline calibration")
        return cls(spec["slope_mm_per_signal_unit"], spec["observable"],
                   DASSettings.from_dict(spec["settings"]), spec["provenance"])


def fit_das_baseline(dataset: BankDataset, settings: DASSettings, record_budgets: tuple[int, ...],
                      *, seed: int = 37, mode: str = "debug", damaged_weight: float = 10.,
                      guard: ResourceGuard | None = None) -> DASDepthCalibration:
    """Through-origin weighted least-squares conversion, fitted on train only.

    This empirical scalar calibration does not turn DAS into a physical depth
    inversion. It makes a reproducible quantitative baseline whose errors can
    be measured honestly. No per-image maximum scaling is used.
    """
    from traxis.config import positive
    positive("damaged weight", damaged_weight)
    entries = dataset.select("train")
    if not entries or len({e.observable for e in entries}) != 1:
        raise ValueError("DAS calibration needs training banks in one signal unit")
    if mode not in ("debug", "research") or (mode == "debug" and len(entries) * len(record_budgets) > 16):
        raise ValueError("debug calibration is limited to sixteen specimen/subset images")
    if mode == "research" and any(e.validation_status == "plumbing_only" or e.observable == "arbitrary" for e in entries):
        raise ValueError("research calibration requires verified physical-unit training data")
    if not record_budgets or any(type(n) is not int or n < 1 for n in record_budgets):
        raise ValueError("calibration budgets must be positive integers")
    numerator, denominator, intact_seen, damaged_seen = 0., 0., False, False
    rng = np.random.default_rng(seed)
    for entry in entries:
        bank = dataset.read(entry)
        intact_seen |= not bank.depth_mm.any()
        damaged_seen |= bool(bank.depth_mm.any())
        truth = bank.depth_mm
        weights = np.where(truth > 0, damaged_weight, 1.)
        weight_sum = float(weights.sum())
        for budget in record_budgets:
            if guard:
                guard.check()
            image = delay_and_sum(subset_state(bank, rng, budget), settings, guard).intensity.astype(float)
            image = resample_cell_map(image, truth.shape)
            numerator += float((weights * image * truth).sum() / weight_sum)
            denominator += float((weights * image**2).sum() / weight_sum)
    if not intact_seen or not damaged_seen or denominator == 0 or numerator <= 0:
        raise ValueError("DAS calibration needs both intact and damaged controls with nonzero damage evidence")
    provenance = {"mode": mode, "dataset_fingerprint": dataset.fingerprint, "seed": seed,
                  "fit_parents": sorted({e.parent_specimen_id for e in entries}), "validation_parents": [],
                  "record_budgets": list(record_budgets), "damaged_weight": damaged_weight,
                  "loss_grid": "bank_reference_cell_averages", "per_image_weight_normalization": True,
                  "method": "nonnegative through-origin weighted least squares"}
    return DASDepthCalibration(numerator / denominator, entries[0].observable, settings, provenance)
