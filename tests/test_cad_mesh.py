import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from traxis.cad_mesh import cad_mesh
from traxis.config import CircularRecess, Material, Plate
from traxis.fem import assemble

HAS_GMSH = importlib.util.find_spec("gmsh") is not None


class CADPreflightTests(unittest.TestCase):
    def test_geometry_budget_is_explicit_and_does_not_weaken_assembly_defaults(self):
        with patch.dict('sys.modules',{'gmsh':None}):
            with self.assertRaisesRegex(MemoryError,'element/assembly'):
                cad_mesh(Plate(),(),.005,max_tetrahedra=250000)
            with self.assertRaisesRegex(MemoryError,'geometry workspace'):
                cad_mesh(Plate(),(),.005,max_tetrahedra=250000,max_geometry_mib=1)
            for budget in (0,-1,float('nan'),float('inf'),2049):
                with self.assertRaises(ValueError):
                    cad_mesh(Plate(),(),.005,max_geometry_mib=budget)


@unittest.skipUnless(HAS_GMSH, "install optional Gmsh for CAD mesh checks")
class CADMeshTests(unittest.TestCase):
    def test_geometry_only_mode_reports_its_separate_workspace_policy(self):
        mesh,report = cad_mesh(Plate(.02,.02,.003),(),.003,max_assembly_mib=1e-6,max_geometry_mib=16,
                              algorithm=10,optimization='Netgen')
        self.assertEqual(report['screen_policy'],'geometry_only')
        self.assertFalse(report['assembly_screen_applied'])
        self.assertLess(report['screened_geometry_mib'],16)
        self.assertEqual(report['actual_geometry_screen_mib'],len(mesh.tetrahedra)/1024)
        self.assertLess(report['retained_volume_relative_error'],1e-10)
        with self.assertRaisesRegex(MemoryError,'element/assembly'):
            cad_mesh(Plate(.02,.02,.003),(),.003,max_assembly_mib=1e-6)

    def test_hxt_controls_the_observed_thin_plate_sliver_stability_case(self):
        from traxis.forward import ForwardSettings, minimum_stable_substeps
        from traxis.scenarios import AcquisitionSettings, SamplingSettings, sample_scenarios
        scenario = sample_scenarios(SamplingSettings(count=1, intact_fraction=1), Plate(.04, .04, .003),
                                    AcquisitionSettings(inset_m=.006, patch_radius_m=.003,
                                                        frequencies_hz=(100000.,), samples=128, cycles=2))[0]
        _, report = cad_mesh(scenario.plate, (), .002, seed=scenario.geometry_seed, algorithm=10, optimization="Netgen")
        self.assertEqual(report["algorithm_3d"], 10)
        self.assertGreater(report["minimum_quality"], .1)
        steps = minimum_stable_substeps(scenario, ForwardSettings(mesh_size_m=.002, mesh_algorithm=10, mesh_optimization="Netgen"))
        self.assertLessEqual(steps, 16)

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
