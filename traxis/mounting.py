"""Explicit mounting realizations for mechanical PZT-equivalent patches."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PZTMounting:
    offsets_xy_m: tuple[tuple[float, float], ...]
    contact_radius_scales: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, 'offsets_xy_m', tuple(tuple(row) for row in self.offsets_xy_m))
        object.__setattr__(self, 'contact_radius_scales', tuple(self.contact_radius_scales))
        offsets, scales = np.asarray(self.offsets_xy_m), np.asarray(self.contact_radius_scales)
        if (offsets.shape not in ((8, 2), (12, 2)) or scales.shape != (len(offsets),)
                or np.iscomplexobj(offsets) or np.iscomplexobj(scales)
                or not np.isfinite(offsets).all() or not np.isfinite(scales).all() or np.any(scales <= 0)):
            raise ValueError('mounting needs eight/twelve finite offsets and positive radius scales')

    def geometry(self, plate, acquisition):
        nominal = acquisition.sensors(plate)
        if len(nominal) != len(self.offsets_xy_m):
            raise ValueError('mounting sensor count does not match acquisition')
        positions = nominal + np.asarray(self.offsets_xy_m)
        radii = acquisition.patch_radius_m*np.asarray(self.contact_radius_scales)
        extent = np.maximum(radii, acquisition.patch_radius_m)
        if (np.any(positions < extent[:, None])
                or np.any(positions > np.array([plate.width_m, plate.height_m])-extent[:, None])):
            raise ValueError('randomized PZT/contact footprint crosses the plate edge')
        distances = np.linalg.norm(positions[:, None]-positions[None], axis=2)
        np.fill_diagonal(distances, np.inf)
        if np.any(distances < extent[:, None]+extent[None]):
            raise ValueError('randomized PZT/contact footprints overlap')
        return positions, radii
