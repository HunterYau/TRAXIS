import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from traxis.config import Material, Plate
from traxis.dispersion import wavenumber_rad_m
from traxis.elmer_reference import elmer_environment
from traxis.elmer_waveguide import condensed_frequencies, quadratic_lamb_case, read_quadratic_nodes, sagittal_projection, verify_elmer_waves
from traxis.resources import ResourceGuard, ResourceLimits


class PeriodicProjectionTests(unittest.TestCase):
    def test_face_relations_and_independent_interior_values(self):
        plate = Plate(.002, .002, .003)
        grid = np.meshgrid(*[np.linspace(0, length, 3) for length in (.002, .002, .003)], indexing='ij')
        nodes = np.column_stack([value.ravel() for value in grid])
        k = 400.
        projection = sagittal_projection(nodes, plate, 1, k)
        self.assertEqual(projection.shape, (81, 24))
        coeff = np.random.default_rng(17).normal(size=24) + 1j * np.arange(24)
        values = (projection @ coeff).reshape(3, 3, 3, 3)
        np.testing.assert_allclose(values[2, :, :, :], values[0, :, :, :] * np.exp(1j * k * plate.width_m))
        np.testing.assert_array_equal(values[:, 2, :, :], values[:, 0, :, :])
        np.testing.assert_array_equal(values[:, :, :, 1], 0)
        # The cell midpoint must not be tied to the boundary plane by a
        # prescribed exponential interpolation.
        left = projection.getrow(3 * 1).indices[0]
        middle = projection.getrow(3 * 10).indices[0]
        self.assertNotEqual(left, middle)

    def test_unmatched_or_moved_periodic_nodes_fail(self):
        plate = Plate(.002, .002, .003)
        grid = np.meshgrid(*[np.linspace(0, length, 3) for length in (.002, .002, .003)], indexing='ij')
        nodes = np.column_stack([value.ravel() for value in grid])
        with self.assertRaisesRegex(ValueError, 'partner'):
            sagittal_projection(nodes[1:], plate, 1, 400.)
        nodes[0, 0] += 1e-5
        with self.assertRaisesRegex(ValueError, 'reference grid'):
            sagittal_projection(nodes, plate, 1, 400.)

    def test_complex_reduction_uses_conjugate_transpose_and_checks_hermiticity(self):
        basis = np.exp(2j * np.pi * np.outer(np.arange(3), np.arange(3)) / 3) / np.sqrt(3)
        frequencies, _ = condensed_frequencies(np.diag([1., 4., 9.]), np.eye(3), sparse.csr_matrix(basis))
        np.testing.assert_allclose(frequencies, np.array([1., 2.]) / (2 * np.pi), rtol=1e-14)
        bad = np.diag([1., 4., 9.])
        bad[0, 1] = 1
        with self.assertRaisesRegex(ArithmeticError, 'Hermitian'):
            condensed_frequencies(bad, np.eye(3), sparse.csr_matrix(basis))


class QuadraticInputTests(unittest.TestCase):
    def test_quadratic_type_and_connectivity_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes = np.column_stack((np.arange(1, 11), np.full(10, -1), np.arange(30).reshape(10, 3)))
            elements = np.array([[1, 1, 510, *range(1, 11)]])
            np.savetxt(root / 'mesh.nodes', nodes)
            np.savetxt(root / 'mesh.elements', elements)
            (root / 'mesh.header').write_text('10 1 0\n1\n510 1\n')
            np.testing.assert_array_equal(read_quadratic_nodes(root), nodes[:, 2:])
            elements[0, 2] = 504
            np.savetxt(root / 'mesh.elements', elements)
            with self.assertRaisesRegex(ValueError, 'quadratic'):
                read_quadratic_nodes(root)

    def test_bad_protocol_is_rejected_before_solver_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard = ResourceGuard()
            with self.assertRaisesRegex(ValueError, 'unique'):
                verify_elmer_waves(root / 'absent', root / 'duplicate', guard, frequencies_hz=(1e5, 1e5))
            with self.assertRaises(ValueError):
                verify_elmer_waves(root / 'absent', root / 'unsupported', guard, frequencies_hz=(1e6,))
            self.assertEqual(list(root.iterdir()), [])


@unittest.skipUnless(os.environ.get('TRAXIS_ELMER_HOME'), 'set TRAXIS_ELMER_HOME for the optional P2 reference test')
class InstalledQuadraticElmerTests(unittest.TestCase):
    def test_real_quadratic_bending_mode_matches_traction_root(self):
        solver, env = elmer_environment(Path(os.environ['TRAXIS_ELMER_HOME']))
        grid = solver.with_name('ElmerGrid' + solver.suffix)
        k = wavenumber_rad_m(100000., .003, 'A0')
        with tempfile.TemporaryDirectory() as directory:
            result = quadratic_lamb_case(solver, grid, env, Path(directory) / 'cell', wavenumber=k,
                                         thickness_m=.003, cells_per_wavelength=16, thickness_cells=4,
                                         material=Material(), guard=ResourceGuard(ResourceLimits(max_seconds=20)))
            self.assertEqual(result['nodes'], 81)
            self.assertEqual(result['condensed_dofs'], 72)
            self.assertLess(abs(result['fundamental_frequencies_hz'][0] / 100000 - 1), .005)
