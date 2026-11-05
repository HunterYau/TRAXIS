from dataclasses import replace
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import splu

from tests.test_elmer_reference import matrix_export
from tests.test_forward import local_scenario
from traxis.config import Material, Plate
from traxis.elmer_operators import load_operators, read_elmer_sparse_operators, save_operators
from traxis.elmer_reference import elmer_environment, operator_sif, run_monitored, write_elmer_mesh
from traxis.elmer_transient import read_scalar_probe, transient_sif
from traxis.elmer_waveforms import ElmerWaveSettings, run_elmer_waveform_case
from traxis.fem import tone_burst
from traxis.geometry import structured_mesh
from traxis.implicit_fem import implicit_simulate
from traxis.resources import ResourceGuard, ResourceLimits


class ImplicitDynamicsTests(unittest.TestCase):
    def test_factor_diagnostics_do_not_materialize_extra_factor_copies(self):
        class StoredFactor:
            def __init__(self, matrix, **kwargs):
                self.actual = splu(matrix, **kwargs)
                self.nnz = self.actual.nnz
            def solve(self, rhs):
                return self.actual.solve(rhs)
            @property
            def L(self):
                raise AssertionError('diagnostics must not materialize L')
            @property
            def U(self):
                raise AssertionError('diagnostics must not materialize U')
        with patch('traxis.implicit_fem.splu', side_effect=StoredFactor):
            result = implicit_simulate([[4., -4.], [-4., 4.]], [[2., .3], [.3, 1.]], 100, .01, np.eye(2),
                                       initial_displacement=[.1, -.05])
        self.assertGreater(result.factor_nonzeros, 0)
        self.assertLess(np.ptp(result.energy_j) / result.energy_j.mean(), 1e-12)

    def test_coupled_consistent_mass_matches_independent_ode_at_second_order(self):
        mass = np.array([[2., .4], [.4, 3.]])
        stiffness = np.array([[20., -3.], [-3., 30.]])
        u0, v0 = np.array([.1, -.05]), np.array([.02, .03])
        def force(t):
            return np.array([1.2 * np.sin(7 * t), -.3 * np.cos(5 * t)])
        def ode(t, state):
            return np.r_[state[2:], np.linalg.solve(mass, force(t) - stiffness @ state[:2])]
        exact = solve_ivp(ode, (0., 1.5), np.r_[u0, v0], method='DOP853', dense_output=True, rtol=1e-12, atol=1e-14)
        self.assertTrue(exact.success)
        errors = []
        for steps in (60, 120, 240):
            result = implicit_simulate(stiffness, mass, steps, 1.5 / steps, np.eye(2), force=force,
                                       initial_displacement=u0, initial_velocity=v0)
            expected = exact.sol(result.time_s)[:2].T
            errors.append(np.linalg.norm(result.displacement_m - expected) / np.linalg.norm(expected))
            self.assertLess(result.maximum_relative_residual, 1e-12)
        for coarse, fine in zip(errors, errors[1:]):
            self.assertGreater(coarse / fine, 3.5)
            self.assertLess(coarse / fine, 4.5)

    def test_unforced_energy_and_stride_with_rigid_and_elastic_motion(self):
        mass = np.array([[2., .3], [.3, 1.]])
        stiffness = np.array([[4., -4.], [-4., 4.]])
        probes = np.array([[1., 0.], [0., 1.], [.4, .6]])
        initial = np.array([.1, -.05])
        result = implicit_simulate(stiffness, mass, 500, .01, probes, initial_displacement=initial,
                                   initial_velocity=[.02, -.03], output_stride=5)
        np.testing.assert_array_equal(result.displacement_m[0], probes @ initial)
        np.testing.assert_allclose(result.time_s, np.arange(101) * .05, atol=1e-15)
        self.assertLess(np.ptp(result.energy_j) / result.energy_j.mean(), 1e-12)

    def test_bad_operators_forces_and_unaligned_steps_fail(self):
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            implicit_simulate([[1., 2.], [0., 1.]], np.eye(2), 2, .1, np.eye(2))
        with self.assertRaisesRegex(ValueError, 'real'):
            implicit_simulate(np.eye(2) * 1j, np.eye(2), 2, .1, np.eye(2))
        with self.assertRaisesRegex(ValueError, 'positive'):
            implicit_simulate(np.eye(2), -np.eye(2), 2, .1, np.eye(2))
        with self.assertRaisesRegex(ValueError, 'vector'):
            implicit_simulate(np.eye(2), np.eye(2), 2, .1, np.eye(2), force=lambda t: [np.nan, 0.])
        with self.assertRaisesRegex(ValueError, 'stride'):
            implicit_simulate(np.eye(2), np.eye(2), 3, .1, np.eye(2), output_stride=2)
        with self.assertRaises(ValueError):
            transient_sif(Material(), dt_s=.1, steps=3, drive_omega=10., probe_node=1,
                           volume_force_n_m3=1., output_stride=2)


class SparseOperatorTests(unittest.TestCase):
    def test_sparse_permutation_and_archive_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_k, expected_m, _ = matrix_export(root)
            stiffness, mass = read_elmer_sparse_operators(root, 4)
            self.assertTrue(sparse.isspmatrix_csr(stiffness))
            np.testing.assert_allclose(stiffness.toarray(), expected_k, rtol=1e-15)
            np.testing.assert_array_equal(mass.toarray(), expected_m)
            save_operators(root / 'operators.npz', stiffness, mass)
            loaded_k, loaded_m = load_operators(root / 'operators.npz')
            np.testing.assert_array_equal(loaded_k.toarray(), expected_k)
            np.testing.assert_array_equal(loaded_m.toarray(), expected_m)

    def test_duplicate_triplets_oversized_header_and_unsigned_pointer_descent_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, entries = matrix_export(root)
            entries[1, :2] = entries[0, :2]
            np.savetxt(root / 'operator_a.dat', entries)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_elmer_sparse_operators(root, 4)
            matrix_export(root)
            np.savetxt(root / 'operator_sizes.dat', [12, 4000001, 4, 1, 12])
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                read_elmer_sparse_operators(root, 4)
            matrix_export(root)
            stiffness, mass = read_elmer_sparse_operators(root, 4)
            save_operators(root / 'operators.npz', stiffness, mass)
            with np.load(root / 'operators.npz', allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            arrays['stiffness_indptr'] = arrays['stiffness_indptr'].astype(np.uint64)
            arrays['stiffness_indptr'][2] = arrays['stiffness_indptr'][1] - 1
            np.savez_compressed(root / 'operators.npz', **arrays)
            with self.assertRaisesRegex(ValueError, 'arrays'):
                load_operators(root / 'operators.npz')


class ReferenceGuardsTests(unittest.TestCase):
    def test_structured_reference_rejects_recesses_and_oversized_grids_before_external_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'must-not-exist'
            with self.assertRaisesRegex(ValueError, 'intact geometry only'):
                run_elmer_waveform_case(local_scenario(), ElmerWaveSettings(mesh_backend='structured'),
                                         Path('absent'), output, ResourceGuard(), state='current')
            with self.assertRaisesRegex(MemoryError, 'structured P2'):
                run_elmer_waveform_case(local_scenario(intact=True), ElmerWaveSettings(mesh_backend='structured', cells=(100, 100, 10)),
                                         Path('absent'), output, ResourceGuard())
            self.assertFalse(output.exists())

    def test_reference_noise_and_step_limits_fail_before_external_work(self):
        scenario = local_scenario(intact=True)
        noisy = replace(scenario, nuisance=replace(scenario.nuisance, relative_noise_std=.01))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard = ResourceGuard()
            with self.assertRaisesRegex(ValueError, 'zero stochastic noise'):
                run_elmer_waveform_case(noisy, ElmerWaveSettings(), root / 'absent', root / 'noise', guard)
            with self.assertRaisesRegex(ValueError, 'internal-step'):
                run_elmer_waveform_case(scenario, ElmerWaveSettings(max_internal_steps=1), root / 'absent', root / 'steps', guard)
            self.assertFalse((root / 'noise').exists())
            self.assertFalse((root / 'steps').exists())


@unittest.skipUnless(os.environ.get('TRAXIS_ELMER_HOME'), 'set TRAXIS_ELMER_HOME for optional implicit/native comparison')
class NativeImplicitTests(unittest.TestCase):
    def test_factorization_only_reuse_and_output_offset_match_full_assembly(self):
        solver, env = elmer_environment(Path(os.environ['TRAXIS_ELMER_HOME']))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traces = []
            force = np.zeros(54)
            force[2] = 1.
            guard = ResourceGuard(ResourceLimits(max_seconds=20, max_rss_mib=512))
            for reuse, stride in ((False, 1), (True, 2)):
                case = root / f'stride-{stride}'
                case.mkdir()
                write_elmer_mesh(structured_mesh(Plate(.04, .03, .003), (2, 2, 1)), case / 'mesh')
                (case / 'transient.sif').write_text(transient_sif(Material(), dt_s=2.5e-7, steps=126,
                    drive_omega=2 * math.pi * 1e5, probe_node=1, nodal_force_n=force,
                    output_stride=stride, reuse_factorization=reuse, burst_cycles=2))
                result = run_monitored([str(solver), 'transient.sif'], case, env, guard, case / 'solver.log')
                self.assertEqual(result['exit_code'], 0)
                key = 'value: displacement 3 at node 1'
                raw = read_scalar_probe(case, expected_times=np.arange(stride, 127, stride) * 2.5e-7, required_names=[key])[key]
                traces.append(raw if reuse else raw[1::2])
            self.assertLess(np.linalg.norm(traces[0] - traces[1]) / np.linalg.norm(traces[0]), 1e-8)

    def test_implicit_sparse_route_matches_actual_native_p2_aperture_trace(self):
        from traxis.quadratic_patch_verification import verify_quadratic_patches
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'native'
            guard = ResourceGuard(ResourceLimits(max_seconds=30, max_rss_mib=512))
            verify_quadratic_patches(Path(os.environ['TRAXIS_ELMER_HOME']), root, guard)
            (root / 'operator.sif').write_text(operator_sif(Material()), encoding='ascii')
            solver, env = elmer_environment(Path(os.environ['TRAXIS_ELMER_HOME']))
            run = run_monitored([str(solver), 'operator.sif'], root, env, guard, root / 'operator.log')
            self.assertEqual(run['exit_code'], 0)
            stiffness, mass = read_elmer_sparse_operators(root, 75)
            with np.load(root / 'aperture-traces.npz', allow_pickle=False) as saved:
                weights, expected = saved['patch_weights'].copy(), saved['displacement_m'][0].copy()
            force, probe = np.zeros((75, 3)), np.zeros((75, 3))
            force[:, 2], probe[:, 2] = weights[0], weights[1]
            result = implicit_simulate(stiffness, mass, 127, 5e-7, probe.ravel()[None],
                                       force=lambda t: force.ravel() * tone_burst(float(t), 1e5, 2))
            self.assertLess(np.linalg.norm(result.displacement_m[:, 0] - expected) / np.linalg.norm(expected), 1e-8)


@unittest.skipUnless(os.environ.get('TRAXIS_ELMER_HOME') and importlib.util.find_spec('gmsh'),
                     'Elmer and Gmsh are required for the optional reference pipeline')
class NativeReferencePipelineTests(unittest.TestCase):
    def test_structured_intact_grid_uses_exact_box_and_screened_p2_node_count(self):
        home = Path(os.environ['TRAXIS_ELMER_HOME'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = ElmerWaveSettings(mesh_backend='structured', cells=(3, 3, 2), substeps_per_output=1)
            scenario = local_scenario(intact=True)
            values, report = run_elmer_waveform_case(scenario, settings, home, root / 'case',
                ResourceGuard(ResourceLimits(max_seconds=30, max_rss_mib=512)))
            self.assertEqual(report['quadratic_nodes'], 7 * 7 * 5)
            self.assertEqual(report['tetrahedra'], 6 * 3 * 3 * 2)
            self.assertEqual(report['linear_mesh']['backend'], 'structured_six_tet')
            self.assertAlmostEqual(report['linear_mesh']['maximum_element_z_span_m'], .0015)
            self.assertEqual(values.shape, (7, 64))
            self.assertGreater(np.linalg.norm(values), 0.)
            self.assertFalse(report['ready_for_training_data'])

    def test_one_transmitter_reference_and_checked_cache_recovery(self):
        home = Path(os.environ['TRAXIS_ELMER_HOME'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = ElmerWaveSettings(mesh_size_m=.006, substeps_per_output=1)
            scenario = local_scenario(intact=True)
            values, report = run_elmer_waveform_case(scenario, settings, home, root / 'first',
                ResourceGuard(ResourceLimits(max_seconds=30, max_rss_mib=512)), cache_dir=root / 'cache')
            self.assertEqual(values.shape, (7, 64))
            self.assertTrue(report['completed'])
            self.assertGreater(report['quadratic_nodes'], 128)
            self.assertFalse(report['ready_for_training_data'])
            with patch('traxis.elmer_waveforms.sys.executable', 'must-not-run-an-implicit-worker'):
                repeated, recovered = run_elmer_waveform_case(scenario, settings, home, root / 'recovered',
                    ResourceGuard(ResourceLimits(max_seconds=30, max_rss_mib=512)), cache_dir=root / 'cache')
            self.assertTrue(recovered['cache_hit'])
            self.assertEqual(recovered['forward_solves'], 0)
            np.testing.assert_array_equal(repeated, values)
