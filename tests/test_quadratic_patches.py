import itertools
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad_vec

from traxis.config import Plate
from traxis.geometry import structured_mesh
from traxis.patches import disk_triangle_quadratic_integrals
from traxis.quadratic_patches import QuadraticTetMesh, TETRAHEDRON_EDGES, load_quadratic_elmer_mesh, quadratic_surface_patch
from traxis.resources import ResourceGuard, ResourceLimits


def quadratic_mesh(cells):
    linear = structured_mesh(Plate(.04, .03, .003), cells)
    points, edges, tets = list(linear.points_m), {}, []
    for corners in linear.tetrahedra:
        mids = []
        for i, j in TETRAHEDRON_EDGES:
            key = tuple(sorted((int(corners[i]), int(corners[j]))))
            if key not in edges:
                edges[key] = len(points)
                points.append((linear.points_m[key[0]] + linear.points_m[key[1]]) / 2)
            mids.append(edges[key])
        tets.append([*corners, *mids])
    return QuadraticTetMesh(np.array(points), np.array(tets), linear.plate)


class QuadraticIntegralTests(unittest.TestCase):
    def test_full_half_and_quarter_disk_second_moments_and_signed_shapes(self):
        examples = (([[-3., -3.], [3., -3.], [0., 4.]],
                     [math.pi, 0., 0., math.pi / 4, 0., math.pi / 4]),
                    ([[0., -2.], [2., 0.], [0., 2.]],
                     [math.pi / 2, 2 / 3, 0., math.pi / 8, 0., math.pi / 8]),
                    ([[0., 0.], [2., 0.], [0., 2.]],
                     [math.pi / 4, 1 / 3, 1 / 3, math.pi / 16, 1 / 8, math.pi / 16]))
        for vertices, expected in examples:
            vertices = np.asarray(vertices)
            for permutation in itertools.permutations(range(3)):
                selected = vertices[list(permutation)]
                nodes = np.vstack((selected, (selected + np.roll(selected, -1, axis=0)) / 2))
                values = disk_triangle_quadratic_integrals(selected, (0., 0.), 1.)
                x, y = nodes.T
                moments = values @ np.column_stack((np.ones(6), x, y, x*x, x*y, y*y))
                np.testing.assert_allclose(moments, expected, rtol=1e-12, atol=1e-13)
        full = disk_triangle_quadratic_integrals(np.array(examples[0][0]), (0., 0.), 1.)
        self.assertLess(full[:3].min(), -.01)
        self.assertTrue((full[3:] > 0).all())

    def test_whole_triangle_si_scaling_translation_and_zero_overlap(self):
        triangle = np.array([[0., 0.], [.1, 0.], [0., .1]])
        np.testing.assert_allclose(disk_triangle_quadratic_integrals(triangle, (0., 0.), 1.),
                                   [0., 0., 0., .005 / 3, .005 / 3, .005 / 3], atol=1e-14)
        triangle = np.array([[-1.3, -.2], [.7, -1.2], [1.4, 1.1]])
        expected = disk_triangle_quadratic_integrals(triangle, (0., 0.), 1.)
        center, radius = np.array([.317, .229]), .003
        actual = disk_triangle_quadratic_integrals(triangle * radius + center, center, radius)
        np.testing.assert_allclose(actual / radius**2, expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(disk_triangle_quadratic_integrals(triangle + 5, (0., 0.), 1.), 0., atol=1e-13)

    def test_general_intersections_match_independent_slice_gaussian_quadrature(self):
        triangles = (np.array([[-1.3, -.2], [.7, -1.2], [1.4, 1.1]]),
                     np.array([[-.3, .8], [2.1, .9], [.2, -.4]]),
                     np.array([[-1.2, 0.], [1.2, .1], [0., .2]]))
        abscissae, gauss_weights = np.polynomial.legendre.leggauss(3)
        for triangle in triangles:
            coefficients = np.linalg.inv(np.column_stack((np.ones(3), triangle)))
            def slice_integral(x):
                intersections = []
                for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
                    if a[0] != b[0] and min(a[0], b[0]) <= x <= max(a[0], b[0]):
                        intersections.append(a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0]))
                if len(intersections) < 2:
                    return np.zeros(6)
                circle_y = math.sqrt(max(0., 1 - x*x))
                lo, hi = max(min(intersections), -circle_y), min(max(intersections), circle_y)
                if hi <= lo:
                    return np.zeros(6)
                y = (hi + lo) / 2 + abscissae * (hi - lo) / 2
                linear = np.column_stack((np.ones(3), np.full(3, x), y)) @ coefficients
                quadratic = np.column_stack((2 * linear**2 - linear,
                                             4 * linear[:, [0, 1, 2]] * linear[:, [1, 2, 0]]))
                return gauss_weights @ quadratic * (hi - lo) / 2
            cuts = [float(x) for x in triangle[:, 0] if -1 < x < 1]
            for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
                delta = b - a
                for root in np.roots([delta @ delta, 2 * a @ delta, a @ a - 1]):
                    if np.isreal(root) and 0 < float(root) < 1:
                        x = float(a[0] + float(root) * delta[0])
                        if -1 < x < 1:
                            cuts.append(x)
            independent, _ = quad_vec(slice_integral, -1., 1., points=sorted(set(cuts)),
                                       epsabs=1e-11, epsrel=1e-11, limit=500)
            np.testing.assert_allclose(disk_triangle_quadratic_integrals(triangle, (0., 0.), 1.),
                                       independent, rtol=2e-9, atol=1e-10)


class QuadraticMeshPatchTests(unittest.TestCase):
    def test_mesh_independent_quadratic_fields_and_edge_node_permutations(self):
        center, radius = np.array([.0103, .0091]), .0032
        direction = np.array([1., 2., 3.]) / math.sqrt(14)
        for count in (1, 2, 4):
            mesh = quadratic_mesh((count, count, 1))
            vector, report = quadratic_surface_patch(mesh, center, radius, direction)
            self.assertLess(report['full_disk_area_relative_error'], 1e-10)
            self.assertLess(report['centroid_error_m'], 1e-13)
            self.assertLess(report['full_disk_second_moment_error_m2'], 1e-16)
            self.assertGreater(report['negative_shape_coefficients'], 0)
            np.testing.assert_allclose(vector.reshape(-1, 3).sum(axis=0), direction, rtol=1e-12)
            x, y = (mesh.points_m[:, :2] - center).T
            field = .2 + 3*x - 4*y + 5*x*x + 7*x*y - 9*y*y
            self.assertAlmostEqual(float(vector @ (field[:, None] * direction).ravel()), .2 - radius**2, places=12)
            tets = mesh.tetrahedra.copy()
            tets[:, :4] = tets[:, [2, 0, 3, 1]]
            tets[:, 4:] = tets[:, [9, 4, 7, 6, 5, 8]]
            permuted = QuadraticTetMesh(mesh.points_m, tets, mesh.plate)
            np.testing.assert_allclose(quadratic_surface_patch(permuted, center, radius, direction)[0], vector,
                                       rtol=1e-10, atol=1e-12)

    def test_clipped_disk_curved_edges_and_nonconforming_midpoints(self):
        mesh = quadratic_mesh((2, 2, 1))
        _, report = quadratic_surface_patch(mesh, (0., .015), .003)
        self.assertTrue(report['edge_clipped'])
        self.assertAlmostEqual(report['area_m2'], math.pi * .003**2 / 2, places=14)
        self.assertAlmostEqual(report['centroid_xy_m'][0], 4 * .003 / (3 * math.pi), places=13)
        moved = mesh.points_m.copy()
        moved[mesh.tetrahedra[0, 4], 0] += 1e-5
        with self.assertRaisesRegex(ValueError, 'straight-edge'):
            QuadraticTetMesh(moved, mesh.tetrahedra, mesh.plate)
        tets = mesh.tetrahedra.copy()
        original = tets[0, 4]
        points = np.vstack((mesh.points_m, mesh.points_m[original]))
        tets[0, 4] = len(points) - 1
        with self.assertRaisesRegex(ValueError, 'conforming'):
            QuadraticTetMesh(points, tets, mesh.plate)

    def test_native_loader_checks_counts_and_solver_edge_ordering(self):
        mesh = quadratic_mesh((1, 1, 1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savetxt(root / 'mesh.nodes', np.column_stack((np.arange(1, len(mesh.points_m) + 1),
                        np.full(len(mesh.points_m), -1), mesh.points_m)), fmt='%.17g')
            elements = np.column_stack((np.arange(1, len(mesh.tetrahedra) + 1), np.ones(len(mesh.tetrahedra)),
                                        np.full(len(mesh.tetrahedra), 510), mesh.tetrahedra + 1))
            np.savetxt(root / 'mesh.elements', elements, fmt='%.17g')
            (root / 'mesh.header').write_text(f'{len(mesh.points_m)} {len(mesh.tetrahedra)} 0\n1\n510 {len(mesh.tetrahedra)}\n')
            actual = load_quadratic_elmer_mesh(root, mesh.plate)
            np.testing.assert_array_equal(actual.tetrahedra, mesh.tetrahedra)
            elements[:, [7, 8]] = elements[:, [8, 7]]
            np.savetxt(root / 'mesh.elements', elements, fmt='%.17g')
            with self.assertRaisesRegex(ValueError, 'edge ordering'):
                load_quadratic_elmer_mesh(root, mesh.plate)


@unittest.skipUnless(os.environ.get('TRAXIS_ELMER_HOME'), 'set TRAXIS_ELMER_HOME for the optional aperture interface check')
class InstalledQuadraticApertureTests(unittest.TestCase):
    def test_actual_signed_patches_reciprocate_and_burst_matches_reference_source(self):
        from traxis.quadratic_patch_verification import verify_quadratic_patches
        with tempfile.TemporaryDirectory() as directory:
            result = verify_quadratic_patches(Path(os.environ['TRAXIS_ELMER_HOME']), Path(directory) / 'checked',
                                               ResourceGuard(ResourceLimits(max_seconds=20, max_rss_mib=512)))
            self.assertTrue(result['passed'])
            self.assertTrue(result['completed'])
            self.assertFalse(result['ready_for_training_data'])
            self.assertLess(result['relative_reciprocity_error'], 1e-8)
            self.assertLess(result['relative_source_interface_error'], 1e-8)
            self.assertTrue(all(row['negative_shape_coefficients'] > 0 for row in result['patches']))
