"""Independent delayed, lossless fixtures; no plate solve or cloud workload.

Motivated by Pelling/Sarradj (2025), arXiv:2506.08870, and
Rezaian/Huang/Duraisamy (2021), arXiv:2109.10408. These tests demonstrate
limitations of finite-record ERA rather than assert a cure for the full plate.
"""
import unittest

import numpy as np
from scipy.linalg import block_diag
from scipy.signal import dlsim

from examples.pzt_era import fit_era as dense_era, reduced_impulse
from examples.pzt_era_matrix_free import fit_era as fft_era


def delayed_lossless_fixture():
    rotations = [np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])
                 for w in np.linspace(.08, 1.4, 4)]
    a = block_diag(*rotations)
    rng = np.random.default_rng(91)
    b, c = rng.normal(size=(8, 1)), rng.normal(size=(2, 8))
    t = np.arange(257) * .001
    drive = np.zeros(len(t)); drive[1] = 1.
    _, response, _ = dlsim((a, b, c, np.zeros((2, 1)), .001), drive)
    delay = 16
    delayed = np.vstack([np.zeros((delay, 2)), response[:-delay]])
    return t, response, delayed, delay


class ResearchAuditTests(unittest.TestCase):
    def test_fft_truncation_matches_independent_published_era_equations(self):
        t, _, h, _ = delayed_lossless_fixture()
        # Construct both block-Hankel matrices independently, without using
        # production hankel(), dimensions(), or the dense ERA implementation.
        markov, rank = h[2:], 6
        rows = len(markov) // 2; cols = len(markov) - rows
        h0 = np.vstack([markov[i:i + cols].T for i in range(rows)])
        h1 = np.vstack([markov[i + 1:i + 1 + cols].T for i in range(rows)])
        u, s, vt = np.linalg.svd(h0, full_matrices=False)
        root = np.diag(np.sqrt(s[:rank]))
        inverse = np.diag(1 / np.sqrt(s[:rank]))
        a = inverse @ u[:, :rank].T @ h1 @ vt[:rank].T @ inverse
        b = root @ vt[:rank, :1]
        c = u[:2, :rank] @ root
        impulse = np.zeros(len(t)); impulse[1] = 1.
        _, expected, _ = dlsim((a, b, c, np.zeros((2, 1)), .001), impulse)
        model, _ = fft_era(t, h, rank, backend='numpy')
        actual, info = reduced_impulse(model, rank, len(t), diagnostic_only=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
        # Correct formulas can produce an unstable truncated model even when
        # the original lossless rotation blocks have bounded responses.
        self.assertGreater(info['spectral_radius'], 1.00001)
        self.assertFalse(info['growth_screen_pass'])
        self.assertGreater(np.linalg.norm(actual - h) / np.linalg.norm(h), .1)

    def test_full_rank_dense_recovers_delayed_lossless_system(self):
        t, _, h, delay = delayed_lossless_fixture()
        model, diagnostic = dense_era(t, h, delay + 8, backend='numpy')
        self.assertEqual(diagnostic['available_order'], delay + 8)
        actual, info = reduced_impulse(model, delay + 8, len(t))
        self.assertTrue(info['growth_screen_pass'])
        np.testing.assert_allclose(actual, h, rtol=1e-9, atol=1e-10)

    def test_exact_known_delay_can_be_retained_outside_small_realization(self):
        t, original, delayed, delay = delayed_lossless_fixture()
        # This fixture's delay is exactly known. No arrival threshold, fitted
        # shift, padded unknown future samples, or production delay removal.
        model, _ = fft_era(t, original, 8, backend='numpy')
        response, info = reduced_impulse(model, 8, len(t))
        restored = np.vstack([np.zeros((delay, 2)), response[:-delay]])
        self.assertTrue(info['growth_screen_pass'])
        np.testing.assert_allclose(restored, delayed, rtol=1e-9, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
