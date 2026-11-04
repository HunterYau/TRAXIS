import itertools
import math
import unittest

import numpy as np

from traxis.config import Plate
from traxis.fem import assemble, surface_patch_with_report
from traxis.geometry import structured_mesh
from traxis.patches import disk_triangle_shape_integrals


class CircularPatchTests(unittest.TestCase):
    def test_known_half_and_quarter_disk_areas_and_first_moments(self):
        for vertices, expected in (([[0, -2], [2, 0], [0, 2]], (math.pi / 2, 2 / 3, 0)),
                                   ([[0, 0], [2, 0], [0, 2]], (math.pi / 4, 1 / 3, 1 / 3))):
            vertices = np.array(vertices, dtype=float)
            for permutation in itertools.permutations(range(3)):
                points = vertices[list(permutation)]
                weights = disk_triangle_shape_integrals(points, (0, 0), 1.)
                np.testing.assert_allclose(np.r_[weights.sum(), weights @ points], expected, rtol=1e-12, atol=1e-13)
                self.assertGreaterEqual(weights.min(), 0)

    def test_inside_outside_enclosing_tangent_and_scaled_translated_cases(self):
        triangle = np.array([[-3., -3.], [3., -3.], [0., 4.]])
        for radius, center in ((1., np.zeros(2)), (.003, np.array([.317, .229]))):
            points = triangle * radius + center
            weights = disk_triangle_shape_integrals(points, center, radius)
            self.assertAlmostEqual(weights.sum() / (math.pi * radius**2), 1, places=12)
            np.testing.assert_allclose(weights @ points / weights.sum(), center, atol=1e-13)
        small = np.array([[0., 0.], [.1, 0.], [0., .1]])
        np.testing.assert_allclose(disk_triangle_shape_integrals(small, (0, 0), 1.), [.005 / 3] * 3, atol=1e-14)
        for points in ([[2., 2.], [3., 2.], [2., 3.]], [[1., -.5], [2., 0.], [1., .5]]):
            self.assertLess(disk_triangle_shape_integrals(points, (0, 0), 1.).sum(), 1e-13)

    def test_full_disk_affine_probe_and_force_moments_are_mesh_independent(self):
        center, radius = (.0103, .0091), .0032
        for n in (2, 4, 7):
            system = assemble(structured_mesh(Plate(.02, .02, .003), (n, n, 2)))
            vector, report = surface_patch_with_report(system, center, radius)
            weights = vector.reshape(-1, 3)[:, 2]
            self.assertLess(report["full_disk_area_relative_error"], 1e-11)
            self.assertLess(report["centroid_error_m"], 1e-14)
            self.assertAlmostEqual(weights.sum(), 1, places=13)
            xyz = system.mesh.points_m
            affine_field = 2 + 3 * xyz[:, 0] - 4 * xyz[:, 1]
            self.assertAlmostEqual(float(weights @ affine_field), 2 + 3 * center[0] - 4 * center[1], places=13)

    def test_edge_clipped_disk_reports_actual_area_and_invalid_triangles_fail(self):
        system = assemble(structured_mesh(Plate(.02, .02, .003), (4, 4, 2)))
        _, report = surface_patch_with_report(system, (0., .01), .003)
        self.assertTrue(report["edge_clipped"])
        self.assertAlmostEqual(report["area_m2"], math.pi * .003**2 / 2, places=14)
        self.assertAlmostEqual(report["centroid_xy_m"][0], 4 * .003 / (3 * math.pi), places=13)
        with self.assertRaises(ValueError):
            disk_triangle_shape_integrals(np.zeros((3, 2)), (0, 0), 1.)

    def test_general_intersections_match_independent_vertical_slice_quadrature(self):
        from scipy.integrate import quad
        triangles = (np.array([[-1.3, -.2], [.7, -1.2], [1.4, 1.1]]),
                     np.array([[-.3, .8], [2.1, .9], [.2, -.4]]),
                     np.array([[-1.2, 0.], [1.2, .1], [0., .2]]))
        for triangle in triangles:
            coefficients = np.linalg.inv(np.column_stack([np.ones(3), triangle]))
            def slice_integral(x, node):
                intersections = []
                for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
                    if a[0] != b[0] and min(a[0], b[0]) <= x <= max(a[0], b[0]):
                        intersections.append(a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0]))
                if len(intersections) < 2:
                    return 0.
                circle_y = math.sqrt(max(0, 1 - x*x))
                lo, hi = max(min(intersections), -circle_y), min(max(intersections), circle_y)
                if hi <= lo:
                    return 0.
                a, b, c = coefficients[:, node]
                return (a + b*x) * (hi - lo) + c * (hi*hi - lo*lo) / 2
            points = sorted(set(x for x in triangle[:, 0] if -1 < x < 1))
            # Thin circular caps can fall between the adaptive quadrature's
            # sample points. Split at every boundary switch, independently
            # solving edge intersections with polynomial roots.
            for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
                delta = b - a
                for root in np.roots([delta @ delta, 2 * a @ delta, a @ a - 1]):
                    if np.isreal(root) and 0 < float(root) < 1:
                        x = float(a[0] + float(root) * delta[0])
                        if -1 < x < 1:
                            points.append(x)
            points = sorted(set(points))
            independent = [quad(slice_integral, -1, 1, args=(i,), points=points,
                                epsabs=1e-11, epsrel=1e-11, limit=200)[0] for i in range(3)]
            np.testing.assert_allclose(disk_triangle_shape_integrals(triangle, (0, 0), 1.), independent,
                                       rtol=2e-9, atol=1e-10)
