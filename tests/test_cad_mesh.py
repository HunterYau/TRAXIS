import importlib.util
import unittest

import numpy as np

from traxis.cad_mesh import cad_mesh
from traxis.config import CircularRecess, Material, Plate
from traxis.fem import assemble

HAS_GMSH = importlib.util.find_spec("gmsh") is not None


@unittest.skipUnless(HAS_GMSH, "install optional Gmsh for CAD mesh checks")
class CADMeshTests(unittest.TestCase):
    def test_intact_mass_and_partial_recess_geometry(self):
        plate = Plate(0.02, 0.02, 0.003)
        mesh, report = cad_mesh(plate, (), 0.002)
        self.assertLess(report["retained_volume_relative_error"], 1e-10)
        system = assemble(mesh)
        self.assertAlmostEqual(system.mass_kg[::3].sum() / (0.02 * 0.02 * 0.003 * Material().density_kg_m3), 1, places=10)
        recess = CircularRecess(0.01, 0.01, 0.004, 0.0015)
        damaged, report = cad_mesh(plate, (recess,), 0.0015)
        self.assertAlmostEqual(report["requested_removed_volume_m3"] / recess.volume_m3, 1, places=10)
        self.assertLess(report["removed_volume_relative_error"], 0.03)
        self.assertTrue(np.isfinite(damaged.points_m).all())
        self.assertFalse(report["ready_for_training_data"])

    def test_geometry_refinement_reduces_volume_error_and_large_mesh_is_screened(self):
        plate = Plate(0.02, 0.02, 0.003)
        recess = CircularRecess(0.01, 0.01, 0.004, 0.0015)
        errors = [cad_mesh(plate, (recess,), size)[1]["removed_volume_relative_error"] for size in (0.003, 0.0015, 0.001)]
        self.assertGreater(errors[0], errors[-1])
        self.assertLess(errors[-1], 0.01)
        with self.assertRaises(MemoryError):
            cad_mesh(Plate(), (CircularRecess(0.25, 0.25, 0.01, 0.002),), 0.0001)
