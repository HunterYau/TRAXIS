"""Mapped GPU allocation guards and explicitly enabled curved-mesh checks."""
from dataclasses import asdict, replace
import os
import unittest
from unittest.mock import patch

import numpy as np

from examples.gpu_mapped_reference import checked_settings
from traxis.config import CircularRecess, Material, Plate
from traxis.gpu_mapped_tensor import build_gpu_mapped, storage_screen
from traxis.mapped_tensor import action, build_system
from traxis.recess_hex import make_recess_mesh
from traxis.resources import ResourceGuard, ResourceLimits


def small_system(order=3, defective=True, angular_cells=1):
    mesh = make_recess_mesh(Plate(.04, .03, .003175), CircularRecess(.021, .014, .008, .0015),
        order, defective=defective, angular_cells=angular_cells)
    return build_system(mesh, Material())


class MappedGPUScreenTests(unittest.TestCase):
    def test_reference_rejects_changed_thickness_and_acceptance_criteria(self):
        plan = dict(kind='gpu_mapped_recess_reference', plate=asdict(Plate(.04, .03, .003175)),
            recess=asdict(CircularRecess(.021, .014, .008, .0015)), material=asdict(Material()),
            order=3, angular_cells=4, defective=True, geometry_order=3, operator_tolerance=1e-10,
            rigid_tolerance=1e-12, modal_error_ratio_interval=[3.8, 4.2], energy_tolerance=1e-8)
        checked_settings(plan)
        for changed in ({'plate':asdict(Plate(.04, .03, .003))}, {'operator_tolerance':1e-4},
                        {'modal_error_ratio_interval':[1., 5.]}, {'order':True}, {'defective':1}):
            with self.assertRaises(ValueError):
                checked_settings({**plan, **changed})

    def test_forecast_includes_curved_geometry_connectivity_and_resident_vectors(self):
        system = small_system()
        screen = storage_screen(system)
        ns, ne, n = len(system.mesh.points_m), len(system.mesh.element_nodes), 64
        expected = ne*n*84 + 4*4*8 + 16*3*ns*8 + 4096
        self.assertEqual(screen['device_storage_screen_mib']*1024**2, expected)
        self.assertTrue(screen['element_index_table_stored'])
        self.assertTrue(screen['inverse_jacobian_per_gll_point'])
        self.assertFalse(screen['dense_element_matrices_stored_on_device'])
        self.assertEqual(screen['stable_dt_s'], system.stable_dt_s)

    def test_memory_and_device_rejections_precede_cuda_import(self):
        system = small_system()
        with patch('builtins.__import__', side_effect=AssertionError('CUDA import attempted')):
            for kwargs in ({'max_device_mib':1e-6}, {'max_host_mib':1e-6}):
                with self.assertRaises(MemoryError):
                    build_gpu_mapped(system, **kwargs)
            for kwargs in ({'device_id':True}, {'device_id':-1}, {'max_device_mib':float('nan')}):
                with self.assertRaises(ValueError):
                    build_gpu_mapped(system, **kwargs)

    def test_invalid_connectivity_geometry_and_mass_are_rejected_before_cuda(self):
        system = small_system()
        ids = system.mesh.element_nodes.copy(); ids[0, 0] = len(system.mesh.points_m)
        bad_mesh = replace(system.mesh, element_nodes=ids)
        inverse = system.inverse_jacobians.copy(); inverse[0, 0, 0, 0] = np.nan
        mass = system.mass_kg.copy(); mass[0] = -1.
        with patch('builtins.__import__', side_effect=AssertionError('CUDA import attempted')):
            for bad in (replace(system, mesh=bad_mesh), replace(system, inverse_jacobians=inverse),
                        replace(system, mass_kg=mass)):
                with self.assertRaises(ValueError):
                    build_gpu_mapped(bad)


@unittest.skipUnless(os.environ.get('TRAXIS_GPU_CHECKS') == '1', 'explicit CUDA checks not enabled')
class MappedGPUActionTests(unittest.TestCase):
    def test_curved_connected_q3_q4_q5_intact_and_recess_actions(self):
        guard = ResourceGuard(ResourceLimits(max_seconds=90, max_rss_mib=1024,
            max_system_memory_percent=85, min_available_mib=2048))
        rng = np.random.default_rng(941)
        for order in (3, 4, 5):
            for defective in (False, True):
                with self.subTest(order=order, defective=defective):
                    cpu = small_system(order, defective, angular_cells=4)
                    gpu = build_gpu_mapped(cpu, guard=guard)
                    state = rng.normal(size=len(cpu.mass_kg))
                    expected = action(cpu, state, guard)
                    actual = gpu.stiffness@state
                    self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected), 1e-11)
                    np.testing.assert_array_equal(gpu.mass_diagonal, cpu.mass_kg)
                    negative = (-.5*gpu.stiffness)@state
                    self.assertLess(np.linalg.norm(negative+.5*expected)/np.linalg.norm(expected), 1e-11)
                    np.testing.assert_array_equal(gpu.stiffness@np.zeros_like(state), 0.)
                    with self.assertRaises(ValueError):
                        gpu.stiffness@np.full_like(state, np.nan)


if __name__ == '__main__':
    unittest.main()
