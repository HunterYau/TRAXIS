"""Material topology, physical volume and fixed-geometry invariants."""
from collections import Counter
import math
import unittest
from unittest.mock import patch

import numpy as np
from numpy.polynomial.legendre import leggauss

from traxis.config import CircularRecess, Plate
from traxis.recess_hex import evaluate_geometry, make_recess_mesh


def volume_and_boundary(mesh):
    nodes, weights = leggauss(8)
    points = np.stack(np.meshgrid(nodes, nodes, nodes, indexing='ij'), axis=-1).reshape(-1, 3)
    volume_weights = np.kron(np.kron(weights, weights), weights)
    volume, minimum = 0., math.inf
    for element in range(len(mesh.element_nodes)):
        _, jac = evaluate_geometry(mesh, element, points)
        det = np.linalg.det(jac)
        minimum = min(minimum, float(det.min()))
        volume += float(volume_weights @ det)
    uv = np.stack(np.meshgrid(nodes, nodes, indexing='ij'), axis=-1).reshape(-1, 2)
    face_weights = np.kron(weights, weights)
    divergence, normal = 0., np.zeros(3)
    areas = Counter()
    for name, element, axis, side in mesh.boundary_faces:
        reference = np.empty((len(uv), 3)); reference[:, axis] = -1 if side == 0 else 1
        reference[:, [i for i in range(3) if i != axis]] = uv
        physical, jac = evaluate_geometry(mesh, element, reference)
        vector = np.cross(jac[:, :, (axis+1) % 3], jac[:, :, (axis+2) % 3]) * (-1 if side == 0 else 1)
        normal += face_weights @ vector
        divergence += float(face_weights @ np.einsum('qi,qi->q', physical, vector))/3
        areas[name] += float(face_weights @ np.linalg.norm(vector, axis=1))
    return volume, minimum, divergence, normal, areas


class RecessHexTests(unittest.TestCase):
    plate = Plate(.04, .03, .003)
    recess = CircularRecess(.021, .014, .008, .0015)

    def test_shared_topology_and_actual_removed_volume(self):
        # The outer annulus has 8 vertices, 12 edges and 4 faces, hence
        # 4*p*p+4*p planar nodes. Its lower p planes remain in the recess.
        for defective, count, dofs, shared in ((False, 18, 1848, 41), (True, 13, 1488, 24)):
            mesh = make_recess_mesh(self.plate, self.recess, 3, defective=defective)
            self.assertEqual(len(mesh.element_nodes), count)
            self.assertEqual(3*len(mesh.points_m), dofs)
            self.assertEqual(mesh.shared_faces, shared)
            self.assertEqual(len(np.unique(mesh.element_nodes)), len(mesh.points_m))
            self.assertLess(mesh.shared_coordinate_error_m, 1e-15)
            volume, minimum, divergence, normal, areas = volume_and_boundary(mesh)
            self.assertGreater(minimum, 0)
            self.assertLess(abs(volume/divergence-1), 1e-12)
            self.assertLess(np.linalg.norm(normal)/(.04*.03), 1e-12)
            self.assertLess(abs(areas['top']/(.04*.03)-1), 1e-12)
            if defective:
                removed = .04*.03*.003-volume
                self.assertGreater(removed, 0)
                self.assertGreater(areas['recess_floor'], 0)
                self.assertGreater(areas['recess_wall'], 0)
                self.assertLess(abs(removed/(areas['recess_floor']*.0015)-1), 1e-12)
            else:
                self.assertLess(abs(volume/(.04*.03*.003)-1), 1e-12)
                self.assertEqual(areas['recess_floor'], 0)
                self.assertEqual(areas['recess_wall'], 0)

    def test_refined_interfaces_and_boundary_counts(self):
        n, radial, layers = 2, (2, 3), (2, 2)
        mesh = make_recess_mesh(self.plate, self.recess, 2, angular_cells=n,
                                radial_cells=radial, layers=layers)
        faces = Counter(name for name, *_ in mesh.boundary_faces)
        disk, ring = n*n+4*radial[0]*n, 4*radial[1]*n
        self.assertEqual(len(mesh.element_nodes), ring*layers[0]+(disk+ring)*layers[1])
        self.assertEqual(faces, dict(top=disk+ring, bottom=ring, recess_floor=disk,
                                    recess_wall=4*n*layers[0], edge=4*n*sum(layers)))
        volume, minimum, divergence, normal, _ = volume_and_boundary(mesh)
        self.assertGreater(minimum, 0)
        self.assertLess(abs(volume/divergence-1), 1e-12)
        self.assertLess(np.linalg.norm(normal)/(.04*.03), 1e-12)

    def test_geometry_is_fixed_when_displacement_degree_changes(self):
        references = np.random.default_rng(91).uniform(-1, 1, (11, 3))
        meshes = [make_recess_mesh(self.plate, self.recess, p, geometry_order=3) for p in (3, 4, 5)]
        for mesh in meshes[1:]:
            np.testing.assert_array_equal(mesh.geometry_points_m, meshes[0].geometry_points_m)
            np.testing.assert_array_equal(mesh.geometry_elements, meshes[0].geometry_elements)
            np.testing.assert_array_equal(mesh.corner_points_m, meshes[0].corner_points_m)
            np.testing.assert_array_equal(mesh.corner_elements, meshes[0].corner_elements)
            for element in range(len(mesh.element_nodes)):
                for value, expected in zip(evaluate_geometry(mesh, element, references),
                                           evaluate_geometry(meshes[0], element, references)):
                    np.testing.assert_array_equal(value, expected)

    def test_circle_shape_refinement_and_polygon_limit(self):
        errors = []
        for order in range(1, 6):
            mesh = make_recess_mesh(self.plate, self.recess, 3, geometry_order=order)
            volume, minimum, _, _, areas = volume_and_boundary(mesh)
            self.assertGreater(minimum, 0)
            represented = areas['recess_floor']
            errors.append(abs(represented/(math.pi*.008**2)-1))
            self.assertLess(abs((.04*.03*.003-volume)/(represented*.0015)-1), 1e-11)
            if order == 1:
                self.assertAlmostEqual(represented/(2*.008**2), 1., places=12)
        for coarse, fine in zip(errors, errors[1:]):
            self.assertLess(fine, coarse)
        self.assertLess(errors[-1], 1e-5)

    def test_jacobian_derivatives_against_finite_differences(self):
        mesh = make_recess_mesh(self.plate, self.recess, 3)
        points = np.array([[.13, -.27, .41], [-.36, .21, -.12]])
        delta = 1e-5
        for element in range(len(mesh.element_nodes)):
            _, jac = evaluate_geometry(mesh, element, points)
            for axis in range(3):
                step = np.eye(3)[axis]*delta
                fd = (evaluate_geometry(mesh, element, points+step)[0]
                      - evaluate_geometry(mesh, element, points-step)[0])/(2*delta)
                np.testing.assert_allclose(jac[:, :, axis], fd, rtol=1e-8, atol=2e-11)

    def test_invalid_and_oversized_meshes_reject_before_allocation(self):
        with patch('numpy.meshgrid', side_effect=AssertionError('allocation attempted')):
            with self.assertRaises(MemoryError):
                make_recess_mesh(Plate(), CircularRecess(.25, .25, .01, .0015), 5,
                                 angular_cells=64, radial_cells=(64, 64))
        for bad in (True, 0, 6, 2.5):
            with self.assertRaises(ValueError):
                make_recess_mesh(self.plate, self.recess, 3, geometry_order=bad)
        with self.assertRaises(ValueError):
            make_recess_mesh(self.plate, CircularRecess(.008, .014, .008, .0015), 3)
        with self.assertRaises(ValueError):
            make_recess_mesh(self.plate, self.recess, 3, layers=(0, 1))


if __name__ == '__main__':
    unittest.main()
