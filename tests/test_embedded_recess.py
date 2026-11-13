"""Closed geometry, conforming patch/exterior joins and native operators."""
from collections import Counter
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from examples.recess_geometry_reference import circle_reference, native_geometry
from examples.recess_operator_reference import compare_native
from tests.test_recess_hex import volume_and_boundary
from traxis.config import CircularRecess, Material, Plate
from traxis.embedded_recess import make_embedded_recess_mesh, storage_screen
from traxis.mapped_patches import circular_patch
from traxis.mapped_tensor import build_system
from traxis.recess_hex import evaluate_geometry
from traxis.resources import ResourceGuard, ResourceLimits


class EmbeddedRecessTests(unittest.TestCase):
    plate = Plate(.08, .06, .003175)
    recess = CircularRecess(.041, .029, .008, .0015)
    controls = dict(angular_cells=2, radial_cells=(1, 1), outer_cell_size_m=.04)

    def test_closed_geometry_volume_and_all_boundary_types(self):
        for defective in (False, True):
            mesh = make_embedded_recess_mesh(self.plate, self.recess, 3, defective=defective,
                                             layers=(2, 1), **self.controls)
            screen = storage_screen(self.plate, self.recess, 3, defective=defective,
                                    layers=(2, 1), **self.controls)
            self.assertEqual(len(mesh.points_m)*3, screen['vector_dofs'])
            self.assertEqual(len(mesh.element_nodes), screen['elements'])
            self.assertEqual(len(np.unique(mesh.element_nodes)), len(mesh.points_m))
            self.assertEqual(mesh.element_nodes.dtype, np.dtype('int32'))
            self.assertLess(mesh.shared_coordinate_error_m, 1e-15)
            volume, jacobian, divergence, normal, areas = volume_and_boundary(mesh)
            area = circle_reference(.008, 3, 2)[0]
            expected = .08*.06*.003175-(area*.0015 if defective else 0.)
            self.assertGreater(jacobian, 0)
            self.assertLess(abs(volume/expected-1), 1e-12)
            self.assertLess(abs(divergence/volume-1), 1e-12)
            self.assertLess(np.linalg.norm(normal)/(.08*.06), 1e-12)
            self.assertLess(abs(areas['top']/(.08*.06)-1), 1e-12)
            counts = Counter(name for name, *_ in mesh.boundary_faces)
            self.assertEqual(counts['top'], screen['planar_cells'])
            self.assertEqual(counts['bottom'], screen['planar_cells']-(screen['disk_cells'] if defective else 0))
            self.assertEqual(counts['recess_floor'], screen['disk_cells'] if defective else 0)
            self.assertEqual(counts['recess_wall'], 4*2*2 if defective else 0)
            self.assertTrue({'top', 'bottom', 'edge'} <= set(counts))

    def test_geometry_and_corners_stay_fixed_across_displacement_orders(self):
        reference = None; sample = np.random.default_rng(63).uniform(-1, 1, (9, 3))
        for order in (3, 4, 5):
            mesh = make_embedded_recess_mesh(self.plate, self.recess, order, **self.controls)
            if reference is not None:
                for name in ('geometry_points_m', 'geometry_elements', 'corner_points_m', 'corner_elements'):
                    np.testing.assert_array_equal(getattr(mesh, name), getattr(reference, name))
                for e in (0, len(mesh.element_nodes)//2, len(mesh.element_nodes)-1):
                    for a, b in zip(evaluate_geometry(mesh, e, sample), evaluate_geometry(reference, e, sample)):
                        np.testing.assert_array_equal(a, b)
            reference = mesh

    def test_physical_disk_crosses_local_patch_and_cartesian_interfaces(self):
        mesh = make_embedded_recess_mesh(self.plate, self.recess, 4, **self.controls)
        x0, y0, x1, y1 = mesh.patch_bounds_m
        for center in ((x0, y0), (x1, self.recess.y_m), (self.recess.x_m, y1), (.006, .006)):
            port = circular_patch(mesh, center, .003)
            self.assertLess(abs(port.weights.sum()-1), 1e-10)
            self.assertLess(np.max(abs(port.weights@mesh.points_m[port.indices, :2]-center))/.003, 1e-10)
            self.assertGreater(port.report['surface_cells'], 0)

    def test_full_size_forecast_and_invalid_inputs_reject_before_construction(self):
        plate = Plate(.5, .5, .003175); recess = CircularRecess(.255, .245, .015, .0015)
        for defective, elements, dofs in ((False, 20768, 4507515), (True, 20528, 4462575)):
            screen = storage_screen(plate, recess, 4, defective=defective)
            self.assertEqual(screen['elements'], elements)
            self.assertEqual(screen['vector_dofs'], dofs)
            self.assertEqual(screen['blocks'], 17)
            self.assertEqual(screen['exterior_cells'], [45, 43, 43, 45])
            self.assertLess(screen['allocation_upper_bytes'], 512*1024**2)
        with patch('traxis.embedded_recess._planar', side_effect=AssertionError('mesh construction attempted')):
            for kwargs in (dict(max_bytes=1), dict(max_cells=1), dict(max_vector_dofs=1)):
                with self.assertRaises(MemoryError): make_embedded_recess_mesh(plate, recess, 4, **kwargs)
            for kwargs in (dict(outer_cell_size_m=0), dict(angular_cells=True),
                           dict(patch_bounds_m=(0., 0., .4, .4)), dict(outer_cell_size_m=1e-9)):
                with self.assertRaises(ValueError): make_embedded_recess_mesh(plate, recess, 4, **kwargs)

    def test_default_patch_adapts_near_a_plate_edge(self):
        recess = CircularRecess(.010, .029, .008, .0015)
        mesh = make_embedded_recess_mesh(self.plate, recess, 3, **self.controls)
        self.assertGreater(mesh.patch_bounds_m[0], 0)
        self.assertLess(mesh.patch_bounds_m[0], recess.x_m-recess.radius_m)
        system = build_system(mesh, Material())
        self.assertGreater(system.screen['minimum_gll_jacobian'], 0)


@unittest.skipUnless(os.environ.get('TRAXIS_HIGHORDER_CHECKS') == '1', 'explicit native embedded references not enabled')
class EmbeddedNativeTests(unittest.TestCase):
    def test_native_geometry_all_columns_and_global_spectrum(self):
        guard = ResourceGuard(ResourceLimits(max_seconds=90, max_rss_mib=768,
            max_system_memory_percent=85, min_available_mib=2048))
        plate = Plate(.08, .06, .003175); recess = CircularRecess(.041, .029, .008, .0015)
        for defective in (False, True):
            mesh = make_embedded_recess_mesh(plate, recess, 2, geometry_order=2, angular_cells=1,
                radial_cells=(1, 1), outer_cell_size_m=.08, defective=defective, guard=guard)
            geometry, _, model = native_geometry(mesh, guard, return_model=True)
            output = Path(os.environ['TMP'])/('embedded-native-recess' if defective else 'embedded-native-intact')
            output.mkdir(); comparison = compare_native(build_system(mesh, Material(), guard), model, geometry, output, guard)
            self.assertTrue(comparison['passed'])
            self.assertIsNotNone(comparison['bounded_full_matrix'])
            self.assertLess(comparison['bounded_full_matrix']['all_column_relative_l2'], 1e-10)


if __name__ == '__main__': unittest.main()
