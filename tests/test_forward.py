from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from traxis.config import CircularRecess, Material, Plate
from traxis.forward import ForwardSettings, generate_waveform_bank
from traxis.scenarios import AcquisitionSettings, NuisanceSettings, Scenario


def local_scenario(intact=False):
    return Scenario("local-intact" if intact else "local-recess", "parent-intact" if intact else "parent-recess", "train", 2026, 44,
                    Plate(0.04, 0.04, 0.003), () if intact else (CircularRecess(0.02, 0.02, 0.01, 0.0015),),
                    AcquisitionSettings(inset_m=0.006, patch_radius_m=0.003, frequencies_hz=(100000.,), samples=64, cycles=2),
                    NuisanceSettings(Material(), 1., (1.,) * 8, (1.,) * 8, relative_noise_std=0.))


class ForwardTests(unittest.TestCase):
    def test_physics_to_bank_timing_grouped_solves_and_cache_recovery(self):
        scenario = local_scenario()
        settings = ForwardSettings(mesh_backend="structured")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "groups"
            bank, report = generate_waveform_bank(scenario, settings, cache_dir=cache)
            self.assertEqual(bank.waveforms.shape, (28, 64))
            self.assertEqual(report["forward_solves"], 14)
            self.assertEqual(report["cache_hits"], 0)
            self.assertGreater(np.linalg.norm(bank.waveforms - bank.baseline) / np.linalg.norm(bank.baseline), 0.001)
            np.testing.assert_allclose(np.diff(bank.time_s), scenario.acquisition.sample_dt_s)
            self.assertEqual(float(bank.depth_mm.max()), 1.5)
            self.assertEqual(bank.metadata.validation_status, "plumbing_only")
            self.assertFalse(report["ready_for_training_data"])
            self.assertEqual(report["internal_steps"], 63 * report["substeps_per_output"])
            with patch("traxis.forward.simulate", side_effect=AssertionError("cached physics was rerun")):
                replay, reused = generate_waveform_bank(scenario, settings, cache_dir=cache)
            self.assertEqual(reused["cache_hits"], 14)
            self.assertEqual(reused["forward_solves"], 0)
            np.testing.assert_array_equal(bank.waveforms, replay.waveforms)
            # Gain/noise changes do not force new unit-load elastic solves.
            variant = replace(scenario, specimen_id="variant", nuisance_seed=45,
                              nuisance=replace(scenario.nuisance, relative_noise_std=0.02))
            noisy, reused = generate_waveform_bank(variant, settings, cache_dir=cache)
            self.assertEqual(reused["cache_hits"], 14)
            self.assertFalse(np.array_equal(noisy.waveforms, bank.waveforms))
            path = next(cache.glob("*.npz"))
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
            arrays["displacement_m"][0, 0] += 1
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "integrity"):
                generate_waveform_bank(scenario, settings, cache_dir=cache)

    def test_intact_control_reuses_one_system_and_has_zero_ideal_residual(self):
        bank, report = generate_waveform_bank(local_scenario(intact=True), ForwardSettings(mesh_backend="structured"))
        self.assertEqual(report["forward_solves"], 7)
        np.testing.assert_array_equal(bank.waveforms, bank.baseline)
        self.assertFalse(bank.depth_mm.any())

    def test_unsupported_physics_unresolved_geometry_and_runtime_are_rejected(self):
        scenario = local_scenario()
        with self.assertRaisesRegex(ValueError, "supports"):
            generate_waveform_bank(replace(scenario, nuisance=replace(scenario.nuisance, support="clamped")))
        with self.assertRaisesRegex(ValueError, "solves"):
            generate_waveform_bank(scenario, ForwardSettings(max_forward_solves=1))
        with self.assertRaisesRegex(ValueError, "internal steps"):
            generate_waveform_bank(scenario, ForwardSettings(mesh_backend="structured", max_internal_steps=1))
        unresolved = replace(scenario, recesses=(CircularRecess(0.02, 0.02, 0.008, 0.0015),))
        with self.assertRaisesRegex(ValueError, "volume error"):
            generate_waveform_bank(unresolved, ForwardSettings(mesh_backend="structured"))
