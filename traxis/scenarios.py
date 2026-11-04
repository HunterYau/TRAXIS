"""Reproducible virtual specimens, nuisance variants and acquisition recipes."""

from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import numpy as np

from traxis.acquisition import action_catalog, sensor_layout, specimen_split
from traxis.config import CircularRecess, Material, Plate, positive
from traxis.storage import canonical_hash, write_json


@dataclass(frozen=True)
class AcquisitionSettings:
    sensor_count: int = 8
    inset_m: float = 0.02
    patch_radius_m: float = 0.005
    frequencies_hz: tuple[float, ...] = (50000., 75000., 100000., 125000., 150000.)
    reciprocal: bool = True
    sample_dt_s: float = 0.5e-6
    samples: int = 2048
    cycles: int = 5

    def __post_init__(self):
        object.__setattr__(self, "frequencies_hz", tuple(float(f) for f in self.frequencies_hz))
        if self.sensor_count not in (8, 12) or type(self.reciprocal) is not bool:
            raise ValueError("choose eight/twelve sensors and an explicit reciprocity assumption")
        for key in ("inset_m", "patch_radius_m", "sample_dt_s"):
            positive(key, getattr(self, key))
        if type(self.samples) is not int or not 64 <= self.samples <= 16384:
            raise ValueError("samples must be an integer in [64,16384]")
        if type(self.cycles) is not int or not 1 <= self.cycles <= 20:
            raise ValueError("tone-burst cycles must be an integer in [1,20]")
        action_catalog(self.sensor_count, self.frequencies_hz, self.reciprocal)
        # A Hann burst is not strictly band-limited. This screens its first
        # main-lobe zeros, not every spectral tail or the complete DAQ bandwidth.
        main_lobe_upper = max(self.frequencies_hz) * (1 + 2 / self.cycles)
        if main_lobe_upper >= 0.5 / self.sample_dt_s:
            raise ValueError("sample rate does not resolve the declared burst main lobe")
        if (self.samples - 1) * self.sample_dt_s < self.cycles / min(self.frequencies_hz):
            raise ValueError("record is shorter than the slowest excitation burst")

    def sensors(self, plate: Plate) -> np.ndarray:
        xy = sensor_layout(plate, self.sensor_count, self.inset_m)
        r = self.patch_radius_m
        if np.any(xy < r) or np.any(xy > [plate.width_m - r, plate.height_m - r]):
            raise ValueError("PZT-equivalent patch would cross the plate edge")
        return xy


@dataclass(frozen=True)
class NuisanceSettings:
    material: Material
    peak_force_n: float
    sensor_gains: tuple[float, ...]
    current_gain_drift: tuple[float, ...]
    relative_noise_std: float = 0.01
    absolute_noise_std_m: float = 0.
    support: str = "free"

    def __post_init__(self):
        object.__setattr__(self, "sensor_gains", tuple(self.sensor_gains))
        object.__setattr__(self, "current_gain_drift", tuple(self.current_gain_drift))
        positive("peak force", self.peak_force_n)
        if len(self.sensor_gains) not in (8, 12) or len(self.current_gain_drift) != len(self.sensor_gains):
            raise ValueError("coupling and drift vectors must match an eight/twelve-PZT layout")
        for value in self.sensor_gains + self.current_gain_drift:
            positive("coupling/drift gain", value)
        for value in (self.relative_noise_std, self.absolute_noise_std_m):
            if not math.isfinite(value) or value < 0:
                raise ValueError("noise standard deviations must be finite and nonnegative")
        if not isinstance(self.support, str) or not self.support.strip():
            raise ValueError("record the support condition")


@dataclass(frozen=True)
class Scenario:
    specimen_id: str
    parent_specimen_id: str
    split: str
    geometry_seed: int
    nuisance_seed: int
    plate: Plate
    recesses: tuple[CircularRecess, ...]
    acquisition: AcquisitionSettings
    nuisance: NuisanceSettings

    def __post_init__(self):
        if not self.specimen_id or not self.parent_specimen_id or self.split not in ("train", "validation", "test", "uncertainty_calibration"):
            raise ValueError("scenario needs specimen/parent IDs and a simulation split")
        if any(type(seed) is not int or seed < 0 for seed in (self.geometry_seed, self.nuisance_seed)):
            raise ValueError("scenario seeds must be nonnegative integers")
        object.__setattr__(self, "recesses", tuple(self.recesses))
        self.acquisition.sensors(self.plate)
        if len(self.nuisance.sensor_gains) != self.acquisition.sensor_count:
            raise ValueError("scenario coupling gains do not match acquisition layout")
        for recess in self.recesses:
            recess.validate(self.plate)
        json.dumps(asdict(self), allow_nan=False)

    @property
    def fingerprint(self):
        return canonical_hash(asdict(self))

    @classmethod
    def from_dict(cls, value: dict):
        data = dict(value)
        data["plate"] = Plate(**data["plate"])
        data["recesses"] = tuple(CircularRecess(**r) for r in data["recesses"])
        data["acquisition"] = AcquisitionSettings(**data["acquisition"])
        nuisance = dict(data["nuisance"])
        nuisance["material"] = Material(**nuisance["material"])
        data["nuisance"] = NuisanceSettings(**nuisance)
        return cls(**data)


@dataclass(frozen=True)
class SamplingSettings:
    count: int = 1000
    seed: int = 2026
    intact_fraction: float = 0.1
    uncertainty_calibration_fraction: float = 0.0
    diameters_m: tuple[float, ...] = (0.02, 0.03, 0.04)
    depths_m: tuple[float, ...] = (0.00075, 0.0015, 0.002)
    edge_clearance_m: float = 0.03
    sensor_clearance_m: float = 0.005
    young_relative_range: float = 0.02
    density_relative_range: float = 0.01
    poisson_absolute_range: float = 0.005
    coupling_relative_range: float = 0.1
    drift_relative_range: float = 0.02
    force_relative_range: float = 0.1
    relative_noise_std: float = 0.01
    support: str = "free"

    def __post_init__(self):
        for key in ("diameters_m", "depths_m"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
            if not getattr(self, key) or len(set(getattr(self, key))) != len(getattr(self, key)):
                raise ValueError("defect size/depth choices must be nonempty and unique")
            for value in getattr(self, key):
                positive(key, value)
        if type(self.count) is not int or not 1 <= self.count <= 100000:
            raise ValueError("scenario count must be an integer in [1,100000]")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("sampling seed must be a nonnegative integer")
        if not math.isfinite(self.intact_fraction) or not 0 <= self.intact_fraction <= 1:
            raise ValueError("intact fraction must lie in [0,1]")
        if not math.isfinite(self.uncertainty_calibration_fraction) or not 0 <= self.uncertainty_calibration_fraction < .7:
            raise ValueError("uncertainty calibration fraction must lie in [0,0.7)")
        for value in (self.edge_clearance_m, self.sensor_clearance_m):
            if not math.isfinite(value) or value < 0:
                raise ValueError("geometric clearances must be finite and nonnegative")
        for key in ("young_relative_range", "density_relative_range", "coupling_relative_range", "drift_relative_range", "force_relative_range"):
            if not math.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) < 1:
                raise ValueError(f"{key} must lie in [0,1)")
        if not math.isfinite(self.poisson_absolute_range) or self.poisson_absolute_range < 0:
            raise ValueError("Poisson variation must be finite and nonnegative")
        if not math.isfinite(self.relative_noise_std) or self.relative_noise_std < 0:
            raise ValueError("noise level must be finite and nonnegative")


def _sample_nuisance(settings: SamplingSettings, acquisition: AcquisitionSettings, seed: int,
                      nominal: Material = Material()) -> NuisanceSettings:
    rng = np.random.default_rng(seed)
    if settings.poisson_absolute_range >= min(nominal.poisson + 1, 0.5 - nominal.poisson):
        raise ValueError("Poisson variation could leave the elastic material's admissible range")
    def factor(radius):
        return float(rng.uniform(1 - radius, 1 + radius))
    material = Material(nominal.young_pa * factor(settings.young_relative_range),
                        nominal.poisson + float(rng.uniform(-settings.poisson_absolute_range, settings.poisson_absolute_range)),
                        nominal.density_kg_m3 * factor(settings.density_relative_range))
    gains = tuple(factor(settings.coupling_relative_range) for _ in range(acquisition.sensor_count))
    drift = tuple(factor(settings.drift_relative_range) for _ in range(acquisition.sensor_count))
    return NuisanceSettings(material, factor(settings.force_relative_range), gains, drift,
                             settings.relative_noise_std, support=settings.support)


def sample_scenarios(settings: SamplingSettings = SamplingSettings(), plate: Plate = Plate(),
                      acquisition: AcquisitionSettings = AcquisitionSettings()) -> tuple[Scenario, ...]:
    sensor_xy = acquisition.sensors(plate)
    intact_count = int(round(settings.count * settings.intact_fraction))
    if intact_count < settings.count and max(settings.depths_m) >= plate.thickness_m:
        raise ValueError("defect depth choices must leave positive thickness")
    if intact_count < settings.count and max(settings.diameters_m) + 2 * settings.edge_clearance_m >= min(plate.width_m, plate.height_m):
        raise ValueError("diameter/edge-clearance choices do not fit plate")
    result = []
    combinations = [(d, depth) for d in settings.diameters_m for depth in settings.depths_m]
    for index in range(settings.count):
        sequence = np.random.SeedSequence([settings.seed, index])
        geometry_seed, nuisance_seed = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in sequence.spawn(2)]
        rng = np.random.default_rng(geometry_seed)
        recesses = ()
        if index >= intact_count:
            diameter, depth = combinations[(index - intact_count) % len(combinations)]
            radius = diameter / 2
            margin = radius + settings.edge_clearance_m
            for _ in range(1000):
                x = float(rng.uniform(margin, plate.width_m - margin))
                y = float(rng.uniform(margin, plate.height_m - margin))
                if np.all(np.linalg.norm(sensor_xy - [x, y], axis=1) > radius + acquisition.patch_radius_m + settings.sensor_clearance_m):
                    break
            else:
                raise ValueError("could not place a recess clear of the sensor footprints")
            recesses = (CircularRecess(x, y, radius, depth),)
        geometry_hash = canonical_hash({"plate": asdict(plate), "recesses": [asdict(r) for r in recesses]})[:10]
        parent = f"sim-{settings.seed}-{index:06d}-{geometry_hash}"
        nuisance = _sample_nuisance(settings, acquisition, nuisance_seed)
        split = specimen_split(parent, settings.seed)
        if split == "train" and settings.uncertainty_calibration_fraction:
            draw = int(canonical_hash(["uncertainty-calibration", settings.seed, parent])[:16], 16) / 2**64
            if draw < settings.uncertainty_calibration_fraction / .7:
                split = "uncertainty_calibration"
        result.append(Scenario(parent + "-v0", parent, split, geometry_seed,
                               nuisance_seed, plate, recesses, acquisition, nuisance))
    return tuple(result)


def nuisance_variant(scenario: Scenario, settings: SamplingSettings, seed: int) -> Scenario:
    """Perturb one virtual plate without changing parent identity or split."""
    nuisance = _sample_nuisance(settings, scenario.acquisition, seed)
    suffix = canonical_hash({"nuisance": asdict(nuisance), "seed": seed})[:12]
    return replace(scenario, specimen_id=f"{scenario.parent_specimen_id}-v{suffix}", nuisance_seed=seed, nuisance=nuisance)


def save_scenario_plan(path: Path, scenarios: tuple[Scenario, ...]) -> dict:
    ids, roles = set(), {}
    strata = Counter()
    for s in scenarios:
        if s.specimen_id in ids or (s.parent_specimen_id in roles and roles[s.parent_specimen_id] != s.split):
            raise ValueError("duplicate scenario ID or parent crossing splits")
        ids.add(s.specimen_id)
        roles[s.parent_specimen_id] = s.split
        label = "intact" if not s.recesses else ";".join(f"D{r.radius_m * 2000:g}_d{r.depth_m * 1000:g}" for r in s.recesses)
        strata[(s.split, label)] += 1
    if not scenarios:
        raise ValueError("scenario plan must be nonempty")
    body = {"schema_version": 1, "kind": "unsolved_scenario_plan", "ready_for_training_data": False,
            "scenarios": [asdict(s) for s in scenarios], "split_counts": dict(Counter(s.split for s in scenarios)),
            "parent_count": len(roles), "parent_split_counts": dict(Counter(roles.values())),
            "stratum_counts": [{"split": split, "geometry": label, "count": n} for (split, label), n in sorted(strata.items())],
            "note": "Recipes only. Frequencies, variations, supports and source/coupling assumptions require empirical validation."}
    body["fingerprint"] = canonical_hash(body)
    if len(json.dumps(body, allow_nan=False).encode()) > 64 * 1024**2:
        raise MemoryError("scenario plan exceeds 64 MiB; use smaller cohorts")
    write_json(path, body)
    return body


def load_scenario_plan(path: Path) -> tuple[Scenario, ...]:
    path = Path(path)
    if path.stat().st_size > 64 * 1024**2:
        raise MemoryError("scenario plan exceeds 64 MiB")
    spec = json.loads(path.read_text(encoding="utf-8-sig"))
    if spec["schema_version"] != 1 or spec["kind"] != "unsolved_scenario_plan":
        raise ValueError("unsupported scenario plan")
    if canonical_hash({k: v for k, v in spec.items() if k != "fingerprint"}) != spec["fingerprint"]:
        raise ValueError("scenario plan fingerprint mismatch")
    scenarios = tuple(Scenario.from_dict(value) for value in spec["scenarios"])
    if not scenarios:
        raise ValueError("scenario plan is empty")
    ids, roles = set(), {}
    for s in scenarios:
        if s.specimen_id in ids or (s.parent_specimen_id in roles and roles[s.parent_specimen_id] != s.split):
            raise ValueError("duplicate scenario or cross-split parent")
        ids.add(s.specimen_id)
        roles[s.parent_specimen_id] = s.split
    return scenarios
