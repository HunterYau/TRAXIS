from dataclasses import replace

import numpy as np

from traxis.acquisition import action_catalog, sensor_layout
from traxis.bank import BankMetadata, WaveformBank
from traxis.config import CircularRecess, Plate
from traxis.geometry import depth_map_mm


def analytic_bank(intact=False, **metadata_changes) -> WaveformBank:
    """Point-scattering fixture for software checks, not elastic training data."""
    plate = Plate()
    sensors = sensor_layout(plate)
    actions = action_catalog(8, (100000.,))
    time = np.arange(1024) * 1e-6
    baseline, waveforms = [], []
    point = np.array([0.303125, 0.203125])
    for a in actions:
        tx, rx = sensors[[a.transmitter, a.receiver]]
        direct_time = np.linalg.norm(tx - rx) / 2500 + 25e-6
        scatter_time = (np.linalg.norm(tx - point) + np.linalg.norm(rx - point)) / 2500 + 25e-6
        def packet(center):
            return np.exp(-0.5 * ((time - center) / 8e-6)**2) * np.cos(2 * np.pi * 100000 * (time - center))
        baseline.append(packet(direct_time))
        waveforms.append(packet(direct_time) + (0 if intact else 2 * packet(scatter_time)))
    metadata = BankMetadata("fixture-1", "parent-1", "train", "simulation", "arbitrary", "control-fixture",
                            "analytic point-scattering fixture", "plumbing_only", 123,
                            {"group_velocity_m_s": 2500, "offset_s": 25e-6})
    metadata = replace(metadata, **metadata_changes)
    recesses = () if intact else (CircularRecess(*point, 0.015, 0.0015),)
    return WaveformBank(metadata, plate, sensors, actions, time, waveforms, baseline,
                        depth_map_mm(plate, recesses, (512, 512), 2))
