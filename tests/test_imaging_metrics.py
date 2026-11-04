import unittest

import numpy as np

from tests.fixtures import analytic_bank
from traxis.bank import ReplayEnvironment
from traxis.config import Plate
from traxis.imaging import DASSettings, delay_and_sum
from traxis.metrics import depth_metrics, ensemble_statistics, resample_cell_map, uncertainty_diagnostics
from traxis.policies import FixedPolicy, RandomPolicy, measurement_budgets


class ImagingTests(unittest.TestCase):
    def test_localizes_independent_point_scattering_fixture(self):
        bank = analytic_bank()
        env = ReplayEnvironment(bank)
        for i in range(len(bank.actions)):
            state = env.reveal(i)
        image = delay_and_sum(state, DASSettings({100000.: 2500.}, 25e-6, (80, 80)))
        y, x = np.unravel_index(image.intensity.argmax(), image.intensity.shape)
        peak = np.array([(x + 0.5) / 80 * 0.5, (y + 0.5) / 80 * 0.5])
        self.assertLess(np.linalg.norm(peak - [0.303125, 0.203125]), 0.009)
        self.assertEqual(image.record_count, 28)
        self.assertAlmostEqual(float(image.intensity.max()), 2, delta=0.05)

    def test_intact_empty_and_out_of_record_paths(self):
        env = ReplayEnvironment(analytic_bank(intact=True))
        empty = delay_and_sum(env.state(), DASSettings({100000.: 2500.}))
        self.assertFalse(empty.intensity.any())
        self.assertFalse(empty.coverage.any())
        intact = delay_and_sum(env.reveal(0), DASSettings({100000.: 2500.}))
        self.assertFalse(intact.intensity.any())
        outside = delay_and_sum(env.state(), DASSettings({100000.: 2500.}, 1.0))
        self.assertFalse(outside.coverage.any())
        self.assertFalse(outside.intensity.any())

    def test_missing_group_speed_fails_instead_of_using_bulk_speed(self):
        state = ReplayEnvironment(analytic_bank()).reveal(0)
        with self.assertRaisesRegex(ValueError, "missing group velocity"):
            delay_and_sum(state, DASSettings({150000.: 5000.}))


class PolicyTests(unittest.TestCase):
    def test_random_reproducibility_and_no_repeats(self):
        bank = analytic_bank()
        def sequence():
            env, policy = ReplayEnvironment(bank), RandomPolicy(7)
            selected = []
            for _ in bank.actions:
                selected.append(policy.choose(env.state()))
                env.reveal(selected[-1])
            with self.assertRaises(ValueError):
                policy.choose(env.state())
            return selected
        self.assertEqual(sequence(), sequence())
        self.assertEqual(len(set(sequence())), len(bank.actions))

    def test_fixed_starts_diagonals_then_opposing_edges(self):
        bank = analytic_bank()
        env, policy = ReplayEnvironment(bank), FixedPolicy()
        pairs = []
        for _ in range(4):
            i = policy.choose(env.state())
            pairs.append((bank.actions[i].transmitter, bank.actions[i].receiver))
            env.reveal(i)
        self.assertEqual(set(pairs[:2]), {(0, 4), (2, 6)})
        self.assertEqual(set(pairs[2:]), {(1, 5), (3, 7)})
        self.assertEqual(measurement_budgets(140), (10, 20, 40, 80, 140))
        self.assertEqual(measurement_budgets(330), (10, 20, 40, 80, 160, 330))


class MetricTests(unittest.TestCase):
    def test_exact_empty_missed_and_false_positive_maps(self):
        plate = Plate(0.004, 0.004, 0.003)
        truth = np.zeros((4, 4))
        empty = depth_metrics(truth, truth, plate)
        self.assertEqual(empty["dice"], 1)
        self.assertIsNone(empty["centroid_error_mm"])
        self.assertIsNone(empty["relative_loss_volume_error"])
        truth[1, 2] = 1.5
        exact = depth_metrics(truth, truth, plate)
        self.assertEqual(exact["mae_mm"], 0)
        self.assertEqual(exact["centroid_error_mm"], 0)
        self.assertEqual(exact["true_loss_volume_mm3"], 1.5)
        miss = depth_metrics(np.zeros((4, 4)), truth, plate)
        self.assertTrue(miss["missed_defect"])
        self.assertIsNone(miss["centroid_error_mm"])
        self.assertEqual(miss["defect_region_mae_mm"], 1.5)
        fp = depth_metrics(truth, np.zeros((4, 4)), plate)
        self.assertEqual(fp["false_positive_area_mm2"], 1)

    def test_shared_grid_preserves_volume_and_detects_small_defect_errors(self):
        coarse = np.arange(16).reshape(4, 4) / 10
        fine = resample_cell_map(coarse, (8, 8))
        self.assertEqual(coarse.mean(), fine.mean())
        np.testing.assert_array_equal(resample_cell_map(fine, (4, 4)), coarse)
        with self.assertRaises(ValueError):
            resample_cell_map(coarse, (7, 7))
        result = depth_metrics(coarse, fine, Plate())
        self.assertEqual(result["mae_mm"], 0)

    def test_known_centroid_and_error_units(self):
        truth, pred = np.zeros((4, 4)), np.zeros((4, 4))
        truth[0, 0], pred[0, 1] = 1., 2.
        result = depth_metrics(pred, truth, Plate(0.004, 0.004, 0.003))
        self.assertEqual(result["centroid_error_mm"], 1)
        self.assertEqual(result["mae_mm"], 3 / 16)
        self.assertEqual(result["rmse_mm"], np.sqrt(5 / 16))
        self.assertEqual(result["loss_volume_error_mm3"], 1)
        self.assertEqual(result["dice"], 0)

    def test_ensemble_spread_and_degenerate_uncertainty(self):
        values = np.array([np.zeros((2, 2)), np.full((2, 2), 2.)])
        mean, std = ensemble_statistics(values)
        np.testing.assert_array_equal(mean, np.ones((2, 2)))
        np.testing.assert_allclose(std, np.sqrt(2))
        report = uncertainty_diagnostics(mean, std, np.ones((2, 2)))
        self.assertEqual(report["coverage_1sigma"], 1)
        self.assertIsNone(report["spread_error_correlation"])
        report = uncertainty_diagnostics(mean, np.zeros((2, 2)), np.zeros((2, 2)))
        self.assertEqual(report["coverage_1_96sigma"], 0)
        with self.assertRaises(ValueError):
            ensemble_statistics(values[:1])
