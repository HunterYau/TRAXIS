import itertools
import math
import unittest

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad

from traxis.polynomial_patches import _unit_disk_monomials, disk_triangle_barycentric_moments, triangle_powers


class PolynomialPatchTests(unittest.TestCase):
    def test_full_and_quarter_disk_moments_through_degree_four(self):
        for triangle, quarter in (([[-3., -3.], [3., -3.], [0., 4.]], False),
                                  ([[0., 0.], [2., 0.], [0., 2.]], True)):
            expected = []
            for a, b in triangle_powers(4):
                value = math.gamma((a+1)/2)*math.gamma((b+1)/2)/(2*(a+b+2)*math.gamma((a+b+2)/2))
                expected.append(value if quarter else (4*value if a % 2 == b % 2 == 0 else 0.))
            for permutation in itertools.permutations(range(3)):
                measured = _unit_disk_monomials(np.array(triangle)[list(permutation)], 4)
                np.testing.assert_allclose(measured, expected, rtol=3e-13, atol=3e-14)

    def test_entire_triangle_dirichlet_integrals_are_scale_and_order_independent(self):
        triangle = np.array([[.1, -.2], [.3, .1], [-.2, .1]])
        area = abs(np.linalg.det(triangle[1:]-triangle[0]))/2
        exact = [2*area*math.factorial(a)*math.factorial(b)/math.factorial(a+b+2) for a, b in triangle_powers(4)]
        for permutation in itertools.permutations(range(3)):
            for scale, center in ((1., np.zeros(2)), (.003, np.array([.317, .229]))):
                measured = disk_triangle_barycentric_moments(triangle[list(permutation)]*scale+center, center, scale, 4)
                np.testing.assert_allclose(measured/scale**2, exact, rtol=2e-11, atol=3e-14)

    def test_cut_triangles_match_independent_vertical_slice_quadrature(self):
        triangles = (np.array([[-1.3, -.2], [.7, -1.2], [1.4, 1.1]]),
                     np.array([[-.3, .8], [2.1, .9], [.2, -.4]]),
                     np.array([[-1.2, 0.], [1.2, .1], [0., .2]]))
        for triangle in triangles:
            affine = np.linalg.inv(np.column_stack([np.ones(3), triangle]))
            def sliced(x, a, b):
                edges = []
                for p, q in zip(triangle, np.roll(triangle, -1, axis=0)):
                    if p[0] != q[0] and min(p[0], q[0]) <= x <= max(p[0], q[0]):
                        edges.append(p[1]+(x-p[0])*(q[1]-p[1])/(q[0]-p[0]))
                if len(edges) < 2:
                    return 0.
                h = math.sqrt(max(0., 1-x*x))
                lo, hi = max(min(edges), -h), min(max(edges), h)
                if hi <= lo:
                    return 0.
                primitive = ((Polynomial([affine[0,1]+affine[1,1]*x, affine[2,1]])**a)
                             *(Polynomial([affine[0,2]+affine[1,2]*x, affine[2,2]])**b)).integ()
                return primitive(hi)-primitive(lo)
            switches = [x for x in triangle[:,0] if -1 < x < 1]
            for p, q in zip(triangle, np.roll(triangle, -1, axis=0)):
                d = q-p
                for t in np.roots([d @ d, 2*p @ d, p @ p-1]):
                    if np.isreal(t) and 0 < float(t) < 1:
                        x = float(p[0]+float(t)*d[0])
                        if -1 < x < 1:
                            switches.append(x)
            independent = [quad(sliced, -1, 1, args=power, points=sorted(set(switches)),
                                epsabs=2e-12, epsrel=2e-11, limit=200)[0] for power in triangle_powers(4)]
            measured = disk_triangle_barycentric_moments(triangle, (0, 0), 1., 4)
            np.testing.assert_allclose(measured, independent, rtol=2e-8, atol=2e-11)

    def test_zero_intersections_and_invalid_inputs(self):
        for triangle in ([[2., 2.], [3., 2.], [2., 3.]], [[1., -.5], [2., 0.], [1., .5]]):
            np.testing.assert_array_equal(disk_triangle_barycentric_moments(triangle, (0, 0), 1., 4), np.zeros(15))
        for degree in (-1, 5, True, 2.5):
            with self.assertRaises(ValueError):
                triangle_powers(degree)
        with self.assertRaises(ValueError):
            disk_triangle_barycentric_moments(np.zeros((3, 2)), (0, 0), 1., 4)
        with self.assertRaises(ValueError):
            disk_triangle_barycentric_moments(np.ones((3, 2), complex), (0, 0), 1., 4)
