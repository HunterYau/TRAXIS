import unittest

import numpy as np

from traxis.acquisition import action_catalog, sensor_layout, specimen_split
from traxis.config import CircularRecess, Material, Plate
from traxis.fem import element_matrices
from traxis.geometry import depth_map_mm, estimate_mesh, structured_mesh


class GeometryTests(unittest.TestCase):
    def test_physical_parameters_reject_nonfinite_or_impossible_values(self):
        for make in (lambda: Plate(thickness_m=0), lambda: Material(poisson=0.5),
                     lambda: Material(young_pa=float("nan")),
                     lambda: CircularRecess(0, 0, -1, 0.001),
                     lambda: CircularRecess(0.1, 0.1, 0.01, 0.003).validate(Plate())):
            with self.assertRaises(ValueError):
                make()

    def test_label_axes_units_and_volume(self):
        p = Plate(width_m=0.2, height_m=0.1)
        r = CircularRecess(0.14, 0.03, 0.01, 0.0015)
        image = depth_map_mm(p, (r,), (256, 512), supersample=4)
        self.assertEqual(image.shape, (256, 512))
        self.assertEqual(float(image.max()), 1.5)
        volume = image.mean() / 1000 * p.width_m * p.height_m
        self.assertAlmostEqual(volume / r.volume_m3, 1, delta=0.003)
        y, x = np.indices(image.shape)
        self.assertAlmostEqual(float((x * image).sum() / image.sum() + 0.5) / 512 * p.width_m, r.x_m, delta=0.0001)
        self.assertAlmostEqual(float((y * image).sum() / image.sum() + 0.5) / 256 * p.height_m, r.y_m, delta=0.0001)

    def test_intact_and_overlapping_recesses(self):
        p = Plate()
        self.assertFalse(depth_map_mm(p, ()).any())
        r1, r2 = CircularRecess(0.25, 0.25, 0.04, 0.001), CircularRecess(0.25, 0.25, 0.02, 0.002)
        image = depth_map_mm(p, (r1, r2))
        self.assertEqual(image[256, 256], 2)
        self.assertLessEqual(image.max(), p.thickness_m * 1000)

    def test_mesh_partitions_volume_and_conforms(self):
        from collections import Counter
        p = Plate(0.02, 0.01, 0.003)
        mesh = structured_mesh(p, (3, 2, 2))
        _, volumes = element_matrices(mesh.points_m[mesh.tetrahedra], Material())
        self.assertAlmostEqual(volumes.sum(), p.width_m * p.height_m * p.thickness_m, delta=1e-19)
        faces = Counter(tuple(sorted(tet[list(face)])) for tet in mesh.tetrahedra
                        for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
        self.assertTrue(all(count in (1, 2) for count in faces.values()))
        for face, count in faces.items():
            if count == 1:
                pts = mesh.points_m[list(face)]
                self.assertTrue(any(np.ptp(pts[:, axis]) < 1e-14 and
                                    (abs(pts[0, axis]) < 1e-14 or abs(pts[0, axis] - limit) < 1e-14)
                                    for axis, limit in enumerate((p.width_m, p.height_m, p.thickness_m))))

    def test_recess_removes_backside_cells_and_compacts_nodes(self):
        p = Plate(0.02, 0.02, 0.003)
        recess = CircularRecess(0.01, 0.01, 0.006, 0.0015)
        mesh = structured_mesh(p, (8, 8, 4), (recess,))
        self.assertGreater(mesh.removed_volume_m3, 0)
        _, volume = element_matrices(mesh.points_m[mesh.tetrahedra], Material())
        self.assertAlmostEqual(volume.sum() + mesh.removed_volume_m3,
                               p.width_m * p.height_m * p.thickness_m, delta=1e-19)
        self.assertEqual(len(np.unique(mesh.tetrahedra)), len(mesh.points_m))
        centers = mesh.points_m[mesh.tetrahedra].mean(axis=1)
        self.assertTrue(np.all(centers[:, 2][np.linalg.norm(centers[:, :2] - [0.01, 0.01], axis=1) < 0.004] >= 0.0015))

    def test_large_mesh_is_rejected_before_allocation(self):
        estimate = estimate_mesh((1000, 1000, 12))
        self.assertGreater(estimate.assembly_bytes_upper, 10**10)
        with self.assertRaises(MemoryError):
            structured_mesh(Plate(), (1000, 1000, 12))


class AcquisitionTests(unittest.TestCase):
    def test_full_scan_counts_and_unique_actions(self):
        frequencies = (50000., 75000., 100000., 125000., 150000.)
        for n, reciprocal, expected in ((8, True, 140), (12, True, 330), (8, False, 280), (12, False, 660)):
            actions = action_catalog(n, frequencies, reciprocal)
            self.assertEqual(len(actions), expected)
            self.assertEqual(len(set(actions)), expected)
            self.assertTrue(all(a.transmitter != a.receiver for a in actions))

    def test_bad_acquisition_parameters(self):
        for frequencies in ((), (1., 1.), (-1.,), (float("nan"),)):
            with self.assertRaises(ValueError):
                action_catalog(8, frequencies)
        with self.assertRaises(ValueError):
            sensor_layout(Plate(), inset_m=0.3)

    def test_layout_positions_and_pair_order(self):
        for n in (8, 12):
            xy = sensor_layout(Plate(), n)
            self.assertEqual(xy.shape, (n, 2))
            self.assertTrue(np.all((xy > 0) & (xy < 0.5)))
        np.testing.assert_allclose(sensor_layout(Plate())[4], [0.48, 0.48])

    def test_splits_do_not_depend_on_waveform_or_input_order(self):
        ids = [f"specimen-{i}" for i in range(1000)]
        forward = {i: specimen_split(i) for i in ids}
        self.assertEqual(forward, {i: specimen_split(i) for i in reversed(ids)})
        counts = {s: list(forward.values()).count(s) for s in ("train", "validation", "test")}
        self.assertTrue(650 < counts["train"] < 750)
        self.assertTrue(100 < counts["validation"] < 200)
        self.assertNotEqual(forward, {i: specimen_split(i, seed=42) for i in ids})
