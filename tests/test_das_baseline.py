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
