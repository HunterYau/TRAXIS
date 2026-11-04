"""Versioned waveform cache and the inference-time reveal boundary."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from traxis.acquisition import Action
from traxis.config import Plate, positive


def _owned_readonly(value, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _digest(array: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(f"{array.dtype.str}:{array.shape}".encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class BankMetadata:
    specimen_id: str
    parent_specimen_id: str
    split: str
    domain: str
    observable: str
    baseline_specimen_id: str
    solver: str
    validation_status: str
    seed: int
    settings: dict

    def __post_init__(self):
        for name in ("specimen_id", "parent_specimen_id", "baseline_specimen_id", "solver"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.domain not in ("simulation", "physical"):
            raise ValueError("domain must be simulation or physical")
        permitted = (("train", "validation", "test", "uncertainty_calibration") if self.domain == "simulation"
                     else ("calibration", "adaptation", "physical_holdout", "uncertainty_calibration"))
        if self.split not in permitted:
            raise ValueError("split is not permitted for this domain")
        if self.observable not in ("displacement_m", "velocity_m_s", "voltage_v", "arbitrary"):
            raise ValueError("observable must specify supported signal units")
        if self.validation_status not in ("plumbing_only", "numerically_verified", "physically_validated"):
            raise ValueError("unknown forward-model validation status")
        if type(self.seed) is not int or not isinstance(self.settings, dict) or not self.settings:
            raise ValueError("record an integer seed and nonempty generation/import settings")
        # This also rejects NaNs and non-serializable provenance before writing.
        json.dumps(asdict(self), allow_nan=False)


@dataclass(frozen=True)
class WaveformBank:
    metadata: BankMetadata
    plate: Plate
    sensors_xy_m: np.ndarray
    actions: tuple[Action, ...]
    time_s: np.ndarray
    waveforms: np.ndarray
    baseline: np.ndarray
    depth_mm: np.ndarray

    def __post_init__(self):
        sensors = _owned_readonly(self.sensors_xy_m, np.float64)
        time = _owned_readonly(self.time_s, np.float64)
        wave = _owned_readonly(self.waveforms, np.float32)
        baseline = _owned_readonly(self.baseline, np.float32)
        depth = _owned_readonly(self.depth_mm, np.float32)
        actions = tuple(self.actions)
        if sensors.ndim != 2 or sensors.shape[1] != 2 or len(sensors) < 2 or not np.isfinite(sensors).all():
            raise ValueError("sensors must have finite shape (n,2)")
        if np.any(sensors < 0) or np.any(sensors > [self.plate.width_m, self.plate.height_m]):
            raise ValueError("sensor coordinates lie outside plate")
        if len(np.unique(sensors, axis=0)) != len(sensors):
            raise ValueError("sensor positions must be distinct")
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("actions must be nonempty and unique")
        for a in actions:
            if not isinstance(a, Action) or type(a.transmitter) is not int or type(a.receiver) is not int:
                raise ValueError("action sensor IDs must be integers")
            if not 0 <= a.transmitter < len(sensors) or not 0 <= a.receiver < len(sensors) or a.transmitter == a.receiver:
                raise ValueError("invalid transmitter/receiver pair")
            positive("action frequency", a.frequency_hz)
        if time.ndim != 1 or len(time) < 3 or not np.isfinite(time).all():
            raise ValueError("time must be a finite vector with at least three samples")
        dt = np.diff(time)
        if dt[0] <= 0 or not np.allclose(dt, dt[0], rtol=1e-7, atol=1e-15):
            raise ValueError("time samples must increase uniformly")
        if max(a.frequency_hz for a in actions) >= 0.5 / dt[0]:
            raise ValueError("excitation carrier is at/above sampling Nyquist frequency")
        if wave.shape != (len(actions), len(time)) or baseline.shape != wave.shape:
            raise ValueError("waveforms and baseline must have shape (actions,time)")
        if not np.isfinite(wave).all() or not np.isfinite(baseline).all():
            raise ValueError("waveforms and baseline must be finite")
        if depth.ndim != 2 or min(depth.shape) < 1 or not np.isfinite(depth).all():
            raise ValueError("depth label must be a finite 2D grid")
        if np.any(depth < 0) or np.any(depth >= self.plate.thickness_m * 1000):
            raise ValueError("labels must represent partial-thickness loss in mm")
        for name, value in (("sensors_xy_m", sensors), ("actions", actions), ("time_s", time),
                            ("waveforms", wave), ("baseline", baseline), ("depth_mm", depth)):
            object.__setattr__(self, name, value)

    def _arrays(self) -> dict[str, np.ndarray]:
        return {"sensors_xy_m": self.sensors_xy_m, "time_s": self.time_s,
                "waveforms": self.waveforms, "baseline": self.baseline, "depth_mm": self.depth_mm}

    def save(self, path: Path, *, overwrite: bool = False) -> None:
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self._arrays()
        manifest = {"schema_version": 1, "metadata": asdict(self.metadata), "plate": asdict(self.plate),
                    "actions": [asdict(a) for a in self.actions],
                    "sha256": {key: _digest(value) for key, value in arrays.items()}}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, manifest=np.array(json.dumps(manifest, allow_nan=False)), **arrays)
            # os.link is an atomic no-replace publication. Temp and target share
            # a filesystem; replacing an existing bank requires explicit opt-in.
            if overwrite:
                os.replace(temporary, path)
            else:
                os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path, max_uncompressed_mib: float = 256) -> "WaveformBank":
        positive("cache size limit", max_uncompressed_mib)
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) != 6 or len({e.filename for e in entries}) != 6:
                raise ValueError("unexpected or duplicate cache entries")
            if sum(e.file_size for e in entries) > max_uncompressed_mib * 1024**2:
                raise MemoryError("waveform bank exceeds uncompressed size limit")
        with np.load(path, allow_pickle=False) as source:
            expected = {"manifest", "sensors_xy_m", "time_s", "waveforms", "baseline", "depth_mm"}
            if set(source.files) != expected:
                raise ValueError("unexpected cache arrays")
            manifest = json.loads(str(source["manifest"].item()))
            if manifest["schema_version"] != 1:
                raise ValueError("unsupported waveform bank schema")
            arrays = {key: source[key] for key in expected - {"manifest"}}
            if {key: _digest(value) for key, value in arrays.items()} != manifest["sha256"]:
                raise ValueError("waveform bank array integrity check failed")
        return cls(BankMetadata(**manifest["metadata"]), Plate(**manifest["plate"]),
                   actions=tuple(Action(**a) for a in manifest["actions"]), **arrays)


def validate_specimen_splits(banks: list[WaveformBank]) -> None:
    """Reject repeat scans/variants of one parent crossing any split."""
    seen = {}
    specimen_ids = set()
    for bank in banks:
        meta = bank.metadata
        if meta.specimen_id in specimen_ids:
            raise ValueError("duplicate specimen/scan ID")
        specimen_ids.add(meta.specimen_id)
        key = (meta.domain, meta.parent_specimen_id)
        if key in seen and seen[key] != meta.split:
            raise ValueError(f"parent specimen {meta.parent_specimen_id} crosses splits")
        seen[key] = meta.split


@dataclass(frozen=True)
class Observation:
    """Only public geometry/settings and already-acquired mechanical signals.

    Masks distinguish missing records from genuinely zero-valued observations.
    No specimen ID, defect geometry, label or forward-solver settings enter here.
    """
    plate: Plate
    observable: str
    sensors_xy_m: np.ndarray
    actions: tuple[Action, ...]
    time_s: np.ndarray
    waveforms: np.ndarray
    baseline: np.ndarray
    observed: np.ndarray

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.observed))


class ReplayEnvironment:
    """An API boundary for leakage prevention, not a Python security sandbox."""

    def __init__(self, bank: WaveformBank):
        self.__bank = bank
        self.__observed = np.zeros(len(bank.actions), dtype=bool)

    def state(self) -> Observation:
        b = self.__bank
        mask = self.__observed
        return Observation(b.plate, b.metadata.observable, _owned_readonly(b.sensors_xy_m), b.actions, _owned_readonly(b.time_s),
                           _owned_readonly(np.where(mask[:, None], b.waveforms, 0)),
                           _owned_readonly(np.where(mask[:, None], b.baseline, 0)), _owned_readonly(mask))

    def reveal(self, action_index: int) -> Observation:
        if type(action_index) is not int or not 0 <= action_index < len(self.__observed):
            raise ValueError("action index is outside catalog")
        if self.__observed[action_index]:
            raise ValueError("record already acquired")
        self.__observed[action_index] = True
        return self.state()

    def reset(self) -> Observation:
        self.__observed[:] = False
        return self.state()
