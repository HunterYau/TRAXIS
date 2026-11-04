import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import psutil

from traxis.config import Plate
from traxis.elmer_reference import elmer_environment, read_elmer_operators, run_monitored, verify_elmer, write_elmer_mesh
from traxis.geometry import structured_mesh
from traxis.resources import ResourceGuard, ResourceLimitError, ResourceLimits


class NativeMeshTests(unittest.TestCase):
    def test_native_mesh_keeps_si_coordinates_and_positive_orientation(self):
        mesh = structured_mesh(Plate(.04, .03, .003), (1, 1, 1))
        mesh.tetrahedra[:, :2] = mesh.tetrahedra[:, 1::-1]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'mesh'
            write_elmer_mesh(mesh, target)
            points = np.loadtxt(target / 'mesh.nodes')[:, 2:]
            elements = np.loadtxt(target / 'mesh.elements', dtype=int)[:, 3:] - 1
            np.testing.assert_array_equal(points, mesh.points_m)
            self.assertTrue((np.linalg.det(points[elements[:, 1:]] - points[elements[:, :1]]) > 0).all())
            self.assertEqual((target / 'mesh.boundary').read_bytes(), b'')
            with self.assertRaises(FileExistsError):
                write_elmer_mesh(mesh, target)

    def test_missing_solver_does_not_create_an_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / 'absent'
            with self.assertRaises(FileNotFoundError):
                elmer_environment(home)
            self.assertFalse(home.exists())


def matrix_export(directory):
    permutation = np.array([3, 1, 4, 2])
    order = (3 * (permutation - 1)[:, None] + np.arange(3)).ravel()
    generator = np.random.default_rng(123)
    basis = generator.normal(size=(12, 12))
    stiffness, mass = basis.T @ basis + np.eye(12), np.diag(np.arange(1., 13.))
    foreign_k, foreign_m = np.empty_like(stiffness), np.empty_like(mass)
    foreign_k[np.ix_(order, order)], foreign_m[np.ix_(order, order)] = stiffness, mass
    rows, columns = np.indices((12, 12))
    entries = np.column_stack((rows.ravel() + 1, columns.ravel() + 1, foreign_k.ravel(), foreign_m.ravel()))
    np.savetxt(directory / 'operator_a.dat', entries)
    np.savetxt(directory / 'operator_perm.dat', np.column_stack((np.arange(1, 5), permutation)))
    np.savetxt(directory / 'operator_sizes.dat', [12, len(entries), 4, 1, 12])
    return stiffness, mass, entries


class MatrixReaderTests(unittest.TestCase):
    def test_nonidentity_node_permutation_preserves_coordinate_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_k, expected_m, _ = matrix_export(root)
            actual_k, actual_m = read_elmer_operators(root, 4)
            np.testing.assert_allclose(actual_k, expected_k, rtol=1e-15)
            np.testing.assert_array_equal(actual_m, expected_m)

    def test_duplicate_fractional_and_nonbijective_exports_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, entries = matrix_export(root)
            entries[1, :2] = entries[0, :2]
            np.savetxt(root / 'operator_a.dat', entries)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_elmer_operators(root, 4)
            _, _, entries = matrix_export(root)
            entries[0, 0] = 1.5
            np.savetxt(root / 'operator_a.dat', entries)
            with self.assertRaisesRegex(ValueError, 'indexes'):
                read_elmer_operators(root, 4)
            matrix_export(root)
            np.savetxt(root / 'operator_perm.dat', [[1, 1], [2, 1], [3, 3], [4, 4]])
            with self.assertRaisesRegex(ValueError, 'bijection'):
                read_elmer_operators(root, 4)


class ExternalProcessTests(unittest.TestCase):
    def test_time_guard_terminates_the_external_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'process.log'
            # A real sleeping process checks termination/reaping, without native
            # solver dependencies or mocked process state.
            command = [getattr(sys, '_base_executable', sys.executable), '-S', '-c',
                       'import os,time; print(os.getpid(),flush=True); time.sleep(60)']
            sample = {'process_rss_mib': 1., 'system_memory_percent': 20., 'available_mib': 10000.,
                      'cpu_temperatures_c': None, 'gpus': [], 'gpu_status': 'disabled'}
            with patch('traxis.resources.snapshot', side_effect=lambda *_: dict(sample)):
                guard = ResourceGuard(ResourceLimits(max_seconds=.6))
                with self.assertRaises(ResourceLimitError):
                    run_monitored(command, root, os.environ.copy(), guard, log)
            pid = int(log.read_text().strip())
            self.assertFalse(psutil.pid_exists(pid))


@unittest.skipUnless(os.environ.get('TRAXIS_ELMER_HOME'), 'set TRAXIS_ELMER_HOME for the optional independent solver test')
class InstalledElmerTests(unittest.TestCase):
    def test_independent_rectangular_and_skewed_3d_operators(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'checked'
            report = verify_elmer(Path(os.environ['TRAXIS_ELMER_HOME']), output,
                                  ResourceGuard(ResourceLimits(max_seconds=20, max_rss_mib=512)))
            self.assertTrue(report['passed'])
            self.assertEqual(len(report['cases']), 2)
            self.assertFalse(report['ready_for_training_data'])
            self.assertEqual(report['training_updates'], 0)
            self.assertTrue(all(row['six_rigid_modes'] for row in report['cases']))
            saved = json.loads((output / 'elmer-report.json').read_text())
            self.assertEqual(saved['fingerprint'], report['fingerprint'])
            self.assertTrue(all('MAIN: Version:' in row['solver_banner'][0] for row in saved['cases']))
