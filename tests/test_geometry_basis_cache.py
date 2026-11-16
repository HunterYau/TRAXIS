"""Interpolation identities and caller isolation for repeated reference grids."""
import unittest

import numpy as np

from traxis.recess_hex import _cached_lagrange_values, lagrange_values
from traxis.tensor_hex import line_rule


class GeometryBasisCacheTests(unittest.TestCase):
    def test_interpolation_and_derivatives_reproduce_polynomials(self):
        x = np.linspace(-1., 1., 73)
        for order in range(1, 6):
            nodes = line_rule(order)[0]
            v, d = lagrange_values(order, x)
            np.testing.assert_allclose(lagrange_values(order, nodes)[0], np.eye(order+1), atol=2e-14)
            for degree in range(order+1):
                np.testing.assert_allclose(v @ nodes**degree, x**degree, atol=3e-14)
                expected = np.zeros_like(x) if degree == 0 else degree*x**(degree-1)
                np.testing.assert_allclose(d @ nodes**degree, expected, atol=8e-14)

    def test_returned_arrays_are_writable_and_do_not_share_storage(self):
        x = np.linspace(-1., 1., 41)
        original = lagrange_values(4, x)
        expected = [a.copy() for a in original]
        self.assertTrue(all(a.flags.f_contiguous and a.flags.writeable for a in original))
        for a in original:
            a[:] = 0
        fresh = lagrange_values(4, x)
        for a, b in zip(fresh, expected):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(all(not np.shares_memory(a,b) for a,b in zip(original,fresh)))

    def test_exact_grid_values_and_degree_are_part_of_the_key(self):
        x = np.linspace(-1., 1., 41)
        a = lagrange_values(4, x)
        x[20] += .001
        b = lagrange_values(4, x)
        self.assertFalse(np.array_equal(a[0], b[0]))
        self.assertEqual(lagrange_values(3, x)[0].shape, (41,4))
        for order in (True, 4., 0, 6):
            with self.assertRaises(ValueError):
                lagrange_values(order, x)
        for points in (np.zeros((2,2)), [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                lagrange_values(4, points)

    def test_large_and_strided_arrays_preserve_interpolation(self):
        x = np.linspace(-1., 1., 2051)
        v, d = lagrange_values(5, x)
        nodes = line_rule(5)[0]
        np.testing.assert_allclose(v @ nodes**5, x**5, atol=3e-14)
        np.testing.assert_allclose(d @ nodes**5, 5*x**4, atol=8e-14)
        a = lagrange_values(5, x[::3])
        np.testing.assert_array_equal(a[0], v[::3])
        np.testing.assert_array_equal(a[1], d[::3])

    def test_cache_has_a_finite_entry_bound(self):
        for offset in range(40):
            lagrange_values(3, np.linspace(-.9,.9,13)+offset/10000.)
        self.assertLessEqual(_cached_lagrange_values.cache_info().currsize, 16)


if __name__ == '__main__':
    unittest.main()
