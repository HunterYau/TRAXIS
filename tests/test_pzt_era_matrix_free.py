"""Small CPU checks; no full-size SVD, plate simulation, or GPU allocation."""
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from scipy.signal import dlsim

from examples.pzt_era import hankel, fit_era as dense_era, reduced_impulse
from examples.pzt_era_matrix_free import HankelOperator, frobenius_norm, fit_era, memory_estimate
from examples.pzt_broadband import reconstruct, paired_metrics, within_target, interior_impulse
from tests.test_pzt_era import fixture, direct
from tests.test_pzt_broadband import simulate


class MatrixFreeEraTests(TestCase):
    def test_independent_dlsim_nonnormal_modes_and_nearly_cancelling_receivers(self):
        # More modes than the original four-state fixture; an independent
        # SciPy state-space solver checks input timing as well as waveforms.
        rng = np.random.default_rng(90210)
        a = np.zeros((16, 16))
        for k, angle in enumerate(np.linspace(.09, 2.4, 8)):
            rho = rng.uniform(.94, .999)
            a[2*k:2*k+2, 2*k:2*k+2] = rho * np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        transform = np.eye(16) + np.triu(rng.normal(scale=.12, size=(16, 16)), 1)
        a = transform @ a @ np.linalg.inv(transform)
        b, top = rng.normal(size=(16, 1)), rng.normal(size=16)
        c = 1e-10 * np.vstack([top, top + 1e-5 * rng.normal(size=16)])
        system = (a, b, c, np.zeros((2, 1)), .001)
        t = np.arange(257) * .001
        impulse = np.zeros(len(t)); impulse[1] = 1.
        _, h, _ = dlsim(system, impulse)
        model, _ = fit_era(t, h, 16, backend='numpy')
        reduced, _ = reduced_impulse(model, 16, len(t))
        for frequency in (5., 10., 20., 40., 80., 120.):
            drive = np.sin(2 * np.pi * frequency * t) * np.sin(np.pi * t / t[-1])**2
            _, expected, _ = dlsim(system, drive)
            score = paired_metrics(expected, reconstruct(reduced, drive))
            self.assertTrue(within_target(score, .0001), score)

    def test_scheduler_uses_matrix_free_builder_and_original_accuracy_target(self):
        from examples import pzt_mor_study as study
        self.assertIs(study.fit_era, fit_era)
        self.assertEqual(study.TARGET, .1)
        self.assertEqual(study.ORDERS, [32, 64, 128, 256, 512, 1024, 2048])
        self.assertEqual(study.FREQUENCIES, [50000, 100000, 150000, 200000, 250000, 300000])
        self.assertEqual(study.BUILD_BACKEND, 'matrix_free_fft_propack_float64')

    def test_fft_products_transpose_and_blocks_match_explicit_hankel(self):
        rng = np.random.default_rng(39)
        for rows, columns, outputs in ((7, 5, 2), (5, 8, 3), (1, 4, 1), (3, 1, 2)):
            data = rng.normal(size=(rows + columns + 2, outputs))
            for shift in (0, 1):
                dense = hankel(data, rows, columns, shift)
                op = HankelOperator(data, rows, columns, shift=shift)
                x = rng.normal(size=(columns, 19))
                y = rng.normal(size=(rows * outputs, 19))
                np.testing.assert_allclose(op @ x, dense @ x, atol=2e-13)
                np.testing.assert_allclose(op.T @ y, dense.T @ y, atol=2e-13)
                np.testing.assert_allclose(op @ x[:, 0], dense @ x[:, 0], atol=2e-13)
                np.testing.assert_allclose(op.rmatvec(y[:, 0]), dense.T @ y[:, 0], atol=2e-13)
                self.assertAlmostEqual(float(y[:, 0] @ (op @ x[:, 0])),
                                       float((op.T @ y[:, 0]) @ x[:, 0]), places=12)
            self.assertAlmostEqual(frobenius_norm(data, rows, columns), np.linalg.norm(hankel(data, rows, columns)), places=12)

    def test_full_rank_fixture_agrees_with_dense_and_independent_drives(self):
        t, h, a, b, c = fixture()
        model, diagnostic = fit_era(t, h, 4, backend='numpy')
        dense, _ = dense_era(t, h, 4, backend='numpy')
        recovered, info = reduced_impulse(model, 4, len(t))
        original, _ = reduced_impulse(dense, 4, len(t))
        self.assertFalse(diagnostic['dense_hankel_allocated'])
        self.assertTrue(diagnostic['singular_spectrum_is_partial'])
        self.assertLess(diagnostic['verified_svd_residual_over_largest_singular_value'], 1e-10)
        np.testing.assert_allclose(recovered, original, atol=1e-12)
        self.assertTrue(info['tail_uses_partial_singular_spectrum'])
        for frequency in (5., 10., 15., 20., 25., 30.):
            drive = np.sin(2 * np.pi * frequency * t) * np.sin(np.pi * t / t[-1])**2
            score = paired_metrics(direct(a, b, c, drive), reconstruct(recovered, drive))
            self.assertTrue(within_target(score, 1e-6), score)

    def test_truncated_model_matches_dense_same_rank_not_full_rank(self):
        t, h, *_ = fixture()
        small, _ = fit_era(t, h, 2, backend='numpy')
        dense, _ = dense_era(t, h, 4, backend='numpy')
        a, _ = reduced_impulse(small, 2, len(t))
        b, _ = reduced_impulse(dense, 2, len(t))
        np.testing.assert_allclose(a, b, atol=2e-12)
        # Tail includes omitted singular values; it is not falsely reported zero.
        _, diagnostic = reduced_impulse(small, 2, len(t))
        self.assertGreater(diagnostic['hankel_relative_frobenius_tail'], .01)
        self.assertFalse(within_target(paired_metrics(h, a), .1))

    def test_small_production_consistent_mass_integrator(self):
        mass = [[2., .4, .1], [.4, 1.5, .2], [.1, .2, 1.]]
        stiffness = [[3100., -900., 300.], [-900., 2400., -350.], [300., -350., 1900.]]
        probes, load = [[.3, 1., -.2], [.3, 1.001, -.199]], [1., -.3, .2]
        n, dt = 128, .001
        full = simulate(mass, stiffness, load, probes, n, dt, interior_impulse(dt))
        model, _ = fit_era(full.time_s, full.displacement_m, 6, backend='numpy')
        reduced, _ = reduced_impulse(model, 6, n + 1)
        drive = np.random.default_rng(6).normal(size=n + 1); drive[0] = 0.
        expected = simulate(mass, stiffness, load, probes, n, dt, lambda t: drive[round(t / dt)])
        score = paired_metrics(expected.displacement_m, reconstruct(reduced, drive))
        self.assertTrue(within_target(score, 1e-5), score)

    def test_does_not_build_dense_hankel_and_reports_solver_failure(self):
        t, h, *_ = fixture()
        with patch('examples.pzt_era.hankel', side_effect=AssertionError('Dense allocation forbidden')):
            fit_era(t, h, 4, backend='numpy')
        with patch('examples.pzt_era_matrix_free.svds', side_effect=np.linalg.LinAlgError('not converged')):
            with self.assertRaises(np.linalg.LinAlgError):
                fit_era(t, h, 4, backend='numpy')

    def test_bad_singular_vectors_are_rejected(self):
        t, h, *_ = fixture()
        with patch('examples.pzt_era_matrix_free.svds', return_value=(np.ones((126, 4)), np.ones(4), np.ones((4, 64)))):
            with self.assertRaises(ArithmeticError):
                fit_era(t, h, 4, backend='numpy')

    def test_weak_retained_direction_must_pass_its_own_residual(self):
        t, _, a, b, c = fixture()
        c = c.copy()
        c[:, 2:] *= 1e-8
        impulse = np.zeros(len(t)); impulse[1] = 1.
        h = direct(a, b, c, impulse)
        matrix = hankel(h[2:], 63, 64)
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        # Rotate a weak right vector into the nullspace. Orthogonality still
        # holds, and scaling all residuals by sigma[0] hides this inaccuracy.
        right = vt[:4].copy()
        angle = 1e-4
        right[3] = np.cos(angle)*vt[3] + np.sin(angle)*vt[4]
        with patch('examples.pzt_era_matrix_free.svds', return_value=(u[:, :4], s[:4], right)):
            with self.assertRaisesRegex(ArithmeticError, 'per-triplet residual'):
                fit_era(t, h, 4, backend='numpy')

    def test_estimate_allocates_no_table_and_keeps_requested_maximum_order(self):
        with patch('numpy.empty', side_effect=AssertionError('No full-size allocations')):
            size = memory_estimate(65537, 2, 2048)
        self.assertEqual(size['hankel_shape'], [65534, 32768])
        self.assertEqual(size['maximum_order'], 2048)
        self.assertFalse(size['dense_hankel_allocated'])
        self.assertLess(size['estimated_device_workspace_bytes'], 1024**3)
        self.assertLess(size['estimated_host_workspace_bytes'], 8 * 1024**3)

    def test_invalid_operator_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            HankelOperator(np.ones((8, 2)) * 1j, 3, 4)
        with self.assertRaises(ValueError):
            HankelOperator(np.ones((4, 2)), 3, 4)
        op = HankelOperator(np.ones((8, 2)), 3, 4)
        with self.assertRaises(ValueError):
            op @ np.full(4, np.nan)
