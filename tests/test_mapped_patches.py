"""Physical force/moment balance and mounting changes on curved top faces."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_forward import local_scenario
from traxis.config import CircularRecess, Plate
from traxis.mapped_patches import circular_patch, scenario_patches
from traxis.mounting import PZTMounting
from traxis.recess_hex import make_recess_mesh
from traxis.tensor_patches import disk_average


class MappedPatchTests(unittest.TestCase):
    plate = Plate(.04, .03, .003175)
    recess = CircularRecess(.021, .014, .008, .0015)

    def test_affine_central_face_reproduces_tensor_polynomial_disk_moments(self):
        for order in (3, 4, 5):
            mesh = make_recess_mesh(self.plate, self.recess, order)
            center = np.array([.021, .014]); radius = .002
            port = circular_patch(mesh, center, radius)
            xy = (mesh.points_m[port.indices, :2]-center)/radius
            for a in range(order+1):
                for b in range(order+1):
                    self.assertLess(abs(port.weights@(xy[:, 0]**a*xy[:, 1]**b)-disk_average(a, b)), 1e-9)

    def test_curved_interfaces_vertices_and_edge_tangent_disks_balance_force_and_moment(self):
        for order in (3, 4, 5):
            mesh = make_recess_mesh(self.plate, self.recess, order, angular_cells=4)
            centers = ((.029, .014), (.021+.008/np.sqrt(2), .014+.008/np.sqrt(2)),
                       (.025, .014), (.003, .003))
            for center in centers:
                with self.subTest(order=order, center=center):
                    direction = np.array([1., -2., 3.]); direction /= np.linalg.norm(direction)
                    port = circular_patch(mesh, center, .003, direction)
                    force = port.vector(len(mesh.points_m)).reshape(3, -1).T
                    np.testing.assert_allclose(force.sum(axis=0), direction, rtol=0, atol=1e-10)
                    torque = np.cross(mesh.points_m, force).sum(axis=0)
                    np.testing.assert_allclose(torque, np.cross([*center, self.plate.thickness_m], direction), rtol=0, atol=1e-11)
                    self.assertLess(port.report['relative_area_error'], 1e-10)
                    self.assertLess(port.report['radius_scaled_first_moment_error'], 1e-10)
                    self.assertTrue(np.all(mesh.points_m[port.indices, 2] == self.plate.thickness_m))
                    self.assertFalse(port.weights.flags.writeable)

    def test_intact_and_recess_ports_have_the_same_physical_top_trace(self):
        values = []
        for defective in (False, True):
            mesh = make_recess_mesh(self.plate, self.recess, 4, angular_cells=4, defective=defective)
            port = circular_patch(mesh, (.029, .014), .003)
            xy = mesh.points_m[port.indices, :2]
            values.append(port.weights@(np.sin(173*xy[:, 0])+np.cos(87*xy[:, 1])))
        self.assertAlmostEqual(*values, places=12)

    def test_state_ports_resolve_actual_remounting_and_do_not_apply_scalar_gains(self):
        scenario = local_scenario(intact=True)
        scenario = replace(scenario, plate=replace(scenario.plate, thickness_m=.003175),
            current_mounting=PZTMounting(((.0004, -.0003),)+((0., 0.),)*7, (1.1,)+(1.,)*7))
        mesh = make_recess_mesh(scenario.plate, CircularRecess(.02, .02, .01, .0015), 3, defective=False)
        baseline = scenario_patches(mesh, scenario, 'intact')
        current = scenario_patches(mesh, scenario, 'current')
        for state, ports in (('intact', baseline), ('current', current)):
            xy, radii = scenario.sensor_geometry(state)
            for i, port in enumerate(ports):
                np.testing.assert_array_equal(port.report['center_xy_m'], xy[i])
                self.assertEqual(port.report['radius_m'], radii[i])
                self.assertAlmostEqual(port.weights.sum(), 1., places=10)
        a, b = baseline[0].vector(len(mesh.points_m)), current[0].vector(len(mesh.points_m))
        self.assertGreater(np.linalg.norm(a-b)/np.linalg.norm(a), 1e-3)
        # Scalar gains belong to drive/readout transfer, not the unit-force port.
        changed = replace(scenario, nuisance=replace(scenario.nuisance, sensor_gains=(1.1,)+(1.,)*7))
        gain_ports = scenario_patches(mesh, changed, 'intact')
        for before, after in zip(baseline, gain_ports):
            np.testing.assert_array_equal(before.indices, after.indices)
            np.testing.assert_array_equal(before.weights, after.weights)

    def test_invalid_geometry_and_preallocation_limits_reject(self):
        mesh = make_recess_mesh(self.plate, self.recess, 3)
        for center, radius in (((.001, .001), .003), ((np.nan, .01), .003), ((.02, .015), -1.)):
            with self.assertRaises(ValueError):
                circular_patch(mesh, center, radius)
        with patch('traxis.mapped_patches.quad_vec', side_effect=AssertionError('integration attempted')):
            with self.assertRaises(MemoryError):
                circular_patch(mesh, (.02, .015), .01, max_faces=1)
            with self.assertRaises(MemoryError):
                circular_patch(mesh, (.02, .015), .003, max_workspace_mib=1e-6)
        scenario = replace(local_scenario(), plate=Plate(.04, .04, .003175))
        wrong = make_recess_mesh(scenario.plate, scenario.recesses[0], 3, defective=True)
        with self.assertRaises(ValueError):
            scenario_patches(wrong, scenario, 'intact')
        surface = mesh.geometry_points_m.copy()
        surface[mesh.geometry_elements[-1, -1], 2] += .0001
        with self.assertRaises(ValueError):
            circular_patch(replace(mesh, geometry_points_m=surface), (.003, .015), .002)


if __name__ == '__main__':
    unittest.main()
