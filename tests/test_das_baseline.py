import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_dataset import fixture_dataset
from traxis.bank import ReplayEnvironment
from traxis.imaging import DASSettings

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from traxis.das_baseline import DASDepthCalibration, fit_das_baseline


@unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is required")
class DASCalibrationTests(unittest.TestCase):
    def test_calibration_objective_uses_reference_cells_and_equal_image_weights(self):
        from dataclasses import replace
        from types import SimpleNamespace
        from unittest.mock import patch
        from tests.fixtures import analytic_bank
        from traxis.dataset import BankDataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = np.zeros((512, 512), np.float32)
            truth[0, 0] = 2
            damaged = replace(analytic_bank(), depth_mm=truth)
            intact = analytic_bank(intact=True, specimen_id="intact", parent_specimen_id="intact")
            damaged.save(root / "damaged.npz")
            intact.save(root / "intact.npz")
            dataset = BankDataset.build([root / "damaged.npz", root / "intact.npz"], root / "dataset.json")
            for size in (128, 256):
                # An independent constant-feature least-squares example has
                # an analytic slope, even for a sub-output-cell defect.
                with patch("traxis.das_baseline.delay_and_sum",
                           return_value=SimpleNamespace(intensity=np.ones((size, size)))):
                    fitted = fit_das_baseline(dataset, DASSettings({100000.: 2500.}, shape=(size, size)), (1,))
                self.assertAlmostEqual(fitted.slope_mm_per_signal_unit, 10 / (512**2 + 9), places=14)

    def test_train_only_scalar_calibration_roundtrip_and_intact_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture_dataset(root, ("train", "train", "validation", "test"))
            fitted = fit_das_baseline(data, DASSettings({100000.: 2500.}, 25e-6), (2, 4))
            self.assertGreater(fitted.slope_mm_per_signal_unit, 0)
            self.assertEqual(fitted.provenance["fit_parents"], ["parent-0", "parent-1"])
            fitted.save(root / "das.json")
            restored = DASDepthCalibration.load(root / "das.json")
            intact = data.read(data.select("train")[1])
            env = ReplayEnvironment(intact)
            state = env.reveal(0)
            self.assertFalse(restored.predict(state).depth_mm.any())
            self.assertEqual(restored.settings, fitted.settings)
            with self.assertRaises(ValueError):
                fit_das_baseline(data, fitted.settings, (100,))

    def test_no_damaged_or_no_intact_training_cases_cannot_calibrate_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture_dataset(directory)
            with self.assertRaisesRegex(ValueError, "intact and damaged"):
                fit_das_baseline(data, DASSettings({100000.: 2500.}, 25e-6), (2,))
