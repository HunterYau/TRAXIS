"""Physical quantities are SI unless a field explicitly ends in _mm."""

from dataclasses import dataclass
import math


def positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Material:
    young_pa: float = 68.9e9
    poisson: float = 0.33
    density_kg_m3: float = 2700.0

    def __post_init__(self):
        positive("Young's modulus", self.young_pa)
        positive("density", self.density_kg_m3)
        if not math.isfinite(self.poisson) or not -1 < self.poisson < 0.5:
            raise ValueError("Poisson ratio must lie between -1 and 0.5")

    @property
    def lame(self) -> tuple[float, float]:
        e, nu = self.young_pa, self.poisson
        return e * nu / ((1 + nu) * (1 - 2 * nu)), e / (2 * (1 + nu))

    @property
    def bulk_speeds_m_s(self) -> tuple[float, float]:
        lam, mu = self.lame
        return math.sqrt((lam + 2 * mu) / self.density_kg_m3), math.sqrt(mu / self.density_kg_m3)


@dataclass(frozen=True)
class Plate:
    width_m: float = 0.5
    height_m: float = 0.5
    thickness_m: float = 0.003

    def __post_init__(self):
        for name, value in vars(self).items():
            positive(name, value)


@dataclass(frozen=True)
class CircularRecess:
    x_m: float
    y_m: float
    radius_m: float
    depth_m: float

    def __post_init__(self):
        positive("radius", self.radius_m)
        positive("depth", self.depth_m)
        if not all(math.isfinite(v) for v in (self.x_m, self.y_m)):
            raise ValueError("recess center must be finite")

    def validate(self, plate: Plate) -> None:
        if self.depth_m >= plate.thickness_m:
            raise ValueError("recess must leave some plate thickness")
        if not (self.radius_m <= self.x_m <= plate.width_m - self.radius_m
                and self.radius_m <= self.y_m <= plate.height_m - self.radius_m):
            raise ValueError("recess must be wholly inside the plate")

    @property
    def volume_m3(self) -> float:
        return math.pi * self.radius_m**2 * self.depth_m
