"""Topological equivalence, count screens and guarded compact extrusion."""
from collections import Counter
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from examples.recess_geometry_reference import native_geometry
from examples.recess_operator_reference import compare_native
from traxis.compact_recess import make_compact_recess_mesh, storage_screen
from traxis.config import CircularRecess, Material, Plate
from traxis.mapped_tensor import action, build_system
from traxis.recess_hex import evaluate_geometry, make_recess_mesh
from traxis.resources import ResourceGuard, ResourceLimits


def correspondence(old, new):
    count = int(old.max())+1
    lower = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    upper = np.full(count, -1, dtype=np.int64)
    np.minimum.at(lower, old.ravel(), new.ravel()); np.maximum.at(upper, old.ravel(), new.ravel())
    np.testing.assert_array_equal(lower, upper)
    np.testing.assert_array_equal(np.sort(lower), np.arange(count))
    return lower


class CompactRecessTests(unittest.TestCase):
    plate = Plate(.04, .03, .003175)
    recess = CircularRecess(.021, .014, .008, .0015)

    def test_exact_topology_correspondence_with_reference_builder(self):
        for p in (1, 2, 3, 4, 5):
            for defective in (False, True):
                with self.subTest(order=p, defective=defective):
                    settings = dict(geometry_order=3, angular_cells=2, radial_cells=(1, 2), layers=(2, 1), defective=defective)
                    old = make_recess_mesh(self.plate, self.recess, p, **settings)
                    new = make_compact_recess_mesh(self.plate, self.recess, p, **settings)
                    permutation = correspondence(old.element_nodes, new.element_nodes)
                    geometry = correspondence(old.geometry_elements, new.geometry_elements)
                    np.testing.assert_allclose(old.points_m, new.points_m[permutation], rtol=0, atol=1e-16)
                    np.testing.assert_allclose(old.geometry_points_m, new.geometry_points_m[geometry], rtol=0, atol=1e-16)
                    np.testing.assert_array_equal(old.cell_faces, new.cell_faces)
                    np.testing.assert_allclose(old.cell_boxes, new.cell_boxes, rtol=0, atol=1e-18)
                    self.assertEqual(set(old.boundary_faces), set(new.boundary_faces))
                    self.assertEqual(old.shared_faces, new.shared_faces)
                    self.assertEqual(new.element_nodes.dtype, np.dtype('int32'))
                    self.assertFalse(new.points_m.flags.writeable)
                    self.assertLess(new.shared_coordinate_error_m, 1e-15)

    def test_geometry_and_operator_are_equivalent_under_node_permutation(self):
        rng = np.random.default_rng(723)
        for defective in (False, True):
            old = make_recess_mesh(self.plate, self.recess, 4, angular_cells=4, defective=defective)
            new = make_compact_recess_mesh(self.plate, self.recess, 4, angular_cells=4, defective=defective)
            perm = correspondence(old.element_nodes, new.element_nodes)
            one, two = build_system(old, Material()), build_system(new, Material(), max_batch_vectors=1)
            state = rng.normal(size=(3, len(old.points_m))); moved = np.empty_like(state); moved[:, perm] = state
            expected = action(one, state.ravel()).reshape(3, -1)
            measured = action(two, moved.ravel()).reshape(3, -1)[:, perm]
            self.assertLess(np.linalg.norm(expected-measured)/np.linalg.norm(expected), 1e-12)
            np.testing.assert_allclose(one.mass_kg.reshape(3, -1), two.mass_kg.reshape(3, -1)[:, perm], rtol=1e-13)
            points = rng.uniform(-1, 1, (9, 3))
            for e in (0, len(new.element_nodes)//2, len(new.element_nodes)-1):
                for a, b in zip(evaluate_geometry(old, e, points), evaluate_geometry(new, e, points)):
                    np.testing.assert_allclose(a, b, rtol=0, atol=1e-16)

    def test_full_size_counts_and_memory_reject_before_construction(self):
        screen = storage_screen(4, 3, 64, (2, 32), (1, 1), False)
        self.assertEqual(screen['elements'], 25600)
        self.assertLess(screen['vector_dofs'], 12000000)
        self.assertFalse(screen['global_node_dictionary'])
        self.assertLess(screen['allocation_upper_bytes'], 512*1024**2)
        with patch('traxis.compact_recess._planar', side_effect=AssertionError('construction attempted')):
            for kwargs in (dict(angular_cells=256, radial_cells=(256, 256)), dict(max_bytes=1), dict(max_vector_dofs=1)):
                with self.assertRaises(MemoryError):
                    make_compact_recess_mesh(self.plate, self.recess, 4, **kwargs)
        with self.assertRaises(ValueError):
            make_compact_recess_mesh(self.plate, self.recess, 4, angular_cells=True)

    def test_declared_cpu_batch_rejects_before_output_allocation(self):
        mesh = make_compact_recess_mesh(self.plate, self.recess, 3)
        system = build_system(mesh, Material(), max_batch_vectors=1)
        with patch('numpy.zeros_like', side_effect=AssertionError('batch allocated')):
            with self.assertRaises(MemoryError): action(system, np.zeros((2, len(system.mass_kg))))
        for value in (0, True, 33):
            with self.assertRaises(ValueError): build_system(mesh, Material(), max_batch_vectors=value)


@unittest.skipUnless(os.environ.get('TRAXIS_HIGHORDER_CHECKS') == '1', 'explicit native compact-mesh reference not enabled')
class CompactNativeTests(unittest.TestCase):
    def test_native_geometry_all_columns_and_spectrum_on_compact_numbering(self):
        guard = ResourceGuard(ResourceLimits(max_seconds=90, max_rss_mib=768, max_system_memory_percent=85, min_available_mib=2048))
        for defective in (False, True):
            mesh = make_compact_recess_mesh(Plate(.04, .03, .003175), CircularRecess(.021, .014, .008, .0015),
                                           3, angular_cells=1, defective=defective, guard=guard)
            geometry, _, model = native_geometry(mesh, guard, return_model=True)
            system = build_system(mesh, Material(), guard)
            output = Path(os.environ['TMP'])/('compact-native-recess' if defective else 'compact-native-intact')
            output.mkdir()
            comparison = compare_native(system, model, geometry, output, guard)
            self.assertTrue(comparison['passed'])
            self.assertLess(comparison['bounded_full_matrix']['all_column_relative_l2'], 1e-10)


if __name__ == '__main__': unittest.main()
