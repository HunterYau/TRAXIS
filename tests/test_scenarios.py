from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from traxis.config import Plate
from traxis.scenarios import AcquisitionSettings, SamplingSettings, load_scenario_plan, nuisance_variant, sample_scenarios, save_scenario_plan


class ScenarioTests(unittest.TestCase):
    def test_reproducible_balanced_geometry_choices_and_controls(self):
        settings = SamplingSettings(count=100)
        a, b = sample_scenarios(settings), sample_scenarios(settings)
        self.assertEqual(a, b)
        self.assertEqual(sum(not s.recesses for s in a), 10)
        combinations = {(s.recesses[0].radius_m * 2, s.recesses[0].depth_m) for s in a if s.recesses}
        self.assertEqual(len(combinations), 9)
        self.assertEqual({s.split for s in a}, {"train", "validation", "test"})
        self.assertNotEqual(a[10].recesses, a[11].recesses)
        self.assertTrue(all(s.nuisance.material.young_pa != 68.9e9 for s in a))

    def test_nuisance_variant_preserves_geometry_parent_and_split(self):
        base = sample_scenarios(SamplingSettings(count=2))[0]
        changed = nuisance_variant(base, SamplingSettings(), 44)
        self.assertEqual(changed.recesses, base.recesses)
        self.assertEqual(changed.parent_specimen_id, base.parent_specimen_id)
        self.assertEqual(changed.split, base.split)
        self.assertNotEqual(changed.specimen_id, base.specimen_id)
        self.assertNotEqual(changed.nuisance, base.nuisance)

    def test_calibration_reservation_only_reassigns_training_parents(self):
        settings = SamplingSettings(count=100)
        original = sample_scenarios(settings)
        reserved = sample_scenarios(replace(settings, uncertainty_calibration_fraction=.2))
        selected = [s for s in reserved if s.split == "uncertainty_calibration"]
        self.assertTrue(selected)
        for a, b in zip(original, reserved):
            self.assertEqual(a.parent_specimen_id, b.parent_specimen_id)
            self.assertEqual(a.recesses, b.recesses)
            self.assertEqual(a.nuisance, b.nuisance)
            if b.split == "uncertainty_calibration":
                self.assertEqual(a.split, "train")
            else:
                self.assertEqual(a.split, b.split)
        self.assertEqual(nuisance_variant(selected[0], settings, 15).split, "uncertainty_calibration")

    def test_plan_roundtrip_fingerprint_and_split_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            scenarios = sample_scenarios(SamplingSettings(count=20))
            path = Path(directory) / "plan.json"
            report = save_scenario_plan(path, scenarios)
            self.assertEqual(load_scenario_plan(path), scenarios)
            self.assertFalse(report["ready_for_training_data"])
            with self.assertRaises(ValueError):
                save_scenario_plan(Path(directory) / "bad.json", (scenarios[0], replace(scenarios[0], specimen_id="other", split="test" if scenarios[0].split != "test" else "train")))
            path.write_text(path.read_text().replace("unsolved_scenario_plan", "changed_scenario_plan"))
            with self.assertRaises(ValueError):
                load_scenario_plan(path)

    def test_layout_counts_and_invalid_sampling(self):
        acquisition = AcquisitionSettings(sensor_count=12)
        self.assertEqual(len(acquisition.sensors(Plate())), 12)
        for make in (lambda: AcquisitionSettings(sample_dt_s=1e-4),
                     lambda: AcquisitionSettings(samples=64),
                     lambda: SamplingSettings(coupling_relative_range=1),
                     lambda: sample_scenarios(SamplingSettings(depths_m=(0.003,))),
                     lambda: AcquisitionSettings(inset_m=0.001).sensors(Plate())):
            with self.assertRaises(ValueError):
                make()

    def test_all_intact_cohort_does_not_require_unused_recess_choices_to_fit(self):
        controls = sample_scenarios(SamplingSettings(count=2, intact_fraction=1), Plate(0.02, 0.02, 0.003),
                                    AcquisitionSettings(inset_m=0.004, patch_radius_m=0.002))
        self.assertTrue(all(not s.recesses for s in controls))
