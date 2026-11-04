import unittest

import numpy as np
from scipy.linalg import eigh

from traxis.config import CircularRecess, Material, Plate
from traxis.fem import assemble, constitutive, element_matrices, simulate, surface_patch, tone_burst
from traxis.geometry import structured_mesh


class ElasticityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.material = Material()
        cls.mesh = structured_mesh(Plate(0.02, 0.02, 0.003), (2, 2, 2))
        cls.system = assemble(cls.mesh, cls.material)

    def test_element_has_six_rigid_modes_and_positive_strain_modes(self):
        xyz = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        ke, volumes = element_matrices(xyz, self.material)
        values = np.linalg.eigvalsh(ke[0])
        self.assertEqual(np.count_nonzero(abs(values) < values.max() * 1e-12), 6)
        self.assertTrue(np.all(values[6:] > 0))
        self.assertAlmostEqual(volumes[0], 1 / 6)
        # Orientation changes may reorder DOFs, but cannot change strain energy.
        reversed_ke, _ = element_matrices(xyz[:, [0, 2, 1, 3]], self.material)
        permutation = np.array([0, 2, 1, 3])[:, None] * 3 + np.arange(3)
        np.testing.assert_allclose(reversed_ke[0], ke[0][np.ix_(permutation.ravel(), permutation.ravel())])

    def test_affine_patch_energy_matches_continuum(self):
        gradient = np.array([[0.002, 0.003, -0.004], [0.005, -0.003, 0.002], [0.001, -0.002, 0.004]])
        u = (self.mesh.points_m @ gradient.T).ravel()
        strain = np.array([gradient[0, 0], gradient[1, 1], gradient[2, 2],
                           gradient[0, 1] + gradient[1, 0], gradient[1, 2] + gradient[2, 1],
                           gradient[0, 2] + gradient[2, 0]])
        volume = 0.02 * 0.02 * 0.003
        expected = 0.5 * volume * strain @ constitutive(self.material) @ strain
        self.assertAlmostEqual((0.5 * u @ self.system.stiffness @ u) / expected, 1, places=12)

    def test_global_rigid_motion_and_mass(self):
        k = self.system.stiffness
        omega = np.array([0.1, -0.3, 0.2])
        u = (np.cross(omega, self.mesh.points_m) + [0.1, -0.5, 0.2]).ravel()
        residual = np.linalg.norm(k @ u) / (np.linalg.norm(k.data) * np.linalg.norm(u))
        self.assertLess(residual, 1e-14)
        self.assertLess(np.linalg.norm((k - k.T).data), np.linalg.norm(k.data) * 1e-14)
        expected_mass = 0.02 * 0.02 * 0.003 * self.material.density_kg_m3
        for axis in range(3):
            self.assertAlmostEqual(self.system.mass_kg[axis::3].sum(), expected_mass, places=14)

    def test_eigenvalue_stability_bound(self):
        values = eigh(self.system.stiffness.toarray(), np.diag(self.system.mass_kg), eigvals_only=True)
        self.assertLessEqual(values.max(), (2 / self.system.stable_dt_s)**2 * (1 + 1e-12))
        with self.assertRaises(ValueError):
            simulate(self.system, 2, self.system.stable_dt_s, np.zeros((1, self.system.ndof)))

    def test_modified_energy_and_second_order_time_convergence(self):
        values, modes = eigh(self.system.stiffness.toarray(), np.diag(self.system.mass_kg))
        mode = modes[:, 10] * 1e-8
        omega = np.sqrt(values[10])
        dt = self.system.stable_dt_s * 0.8
        steps = 80
        errors = []
        for refinement in (1, 2, 4):
            run = simulate(self.system, steps * refinement, dt / refinement, mode[None, :], initial_displacement=mode)
            exact = mode * np.cos(omega * steps * dt)
            errors.append(np.linalg.norm(run.final_displacement_m - exact))
            self.assertLess(np.ptp(run.modified_energy_j) / run.modified_energy_j[0], 1e-11)
        self.assertGreater(errors[0] / errors[1], 3.8)
        self.assertGreater(errors[1] / errors[2], 3.8)

    def test_zero_state_remains_zero(self):
        run = simulate(self.system, 10, self.system.stable_dt_s / 2, np.zeros((1, self.system.ndof)))
        self.assertFalse(run.final_displacement_m.any())
        self.assertFalse(run.physical_energy_j.any())

    def test_mechanical_source_receiver_reciprocity(self):
        a = surface_patch(self.system, (0.003, 0.003), 0.005)
        b = surface_patch(self.system, (0.017, 0.017), 0.005)
        self.assertAlmostEqual(a.sum(), 1)
        self.assertAlmostEqual(b.sum(), 1)
        dt = self.system.stable_dt_s * 0.8
        r1 = simulate(self.system, 120, dt, b[None, :], force=lambda t: a * tone_burst(t, 100000, 2))
        r2 = simulate(self.system, 120, dt, a[None, :], force=lambda t: b * tone_burst(t, 100000, 2))
        self.assertGreater(np.linalg.norm(r1.displacement_m), 0)
        np.testing.assert_allclose(r1.displacement_m, r2.displacement_m, atol=1e-22, rtol=1e-10)

    def test_backside_removal_changes_response(self):
        damaged = assemble(structured_mesh(self.mesh.plate, (4, 4, 4),
                            (CircularRecess(0.01, 0.01, 0.006, 0.0015),)))
        intact = assemble(structured_mesh(self.mesh.plate, (4, 4, 4)))
        self.assertLess(damaged.mass_kg.sum(), intact.mass_kg.sum())
        traces = []
        dt = min(damaged.stable_dt_s, intact.stable_dt_s) * 0.8
        for system in (intact, damaged):
            a = surface_patch(system, (0.003, 0.01), 0.004)
            b = surface_patch(system, (0.017, 0.01), 0.004)
            run = simulate(system, 250, dt, b[None, :], force=lambda t: a * tone_burst(t, 150000, 2))
            traces.append(run.displacement_m)
        self.assertGreater(np.linalg.norm(traces[0] - traces[1]) / np.linalg.norm(traces[0]), 0.01)

    def test_mesh_refinement_rayleigh_quotient(self):
        # Independent continuum mode: nu=0, u_x=cos(pi*x/L) in a free cube.
        material, plate = Material(poisson=0), Plate(0.01, 0.01, 0.01)
        exact = material.young_pa / material.density_kg_m3 * (np.pi / plate.width_m)**2
        errors = []
        for n in (2, 4, 8):
            system = assemble(structured_mesh(plate, (n, n, n)), material)
            u = np.zeros((len(system.mesh.points_m), 3))
            u[:, 0] = np.cos(np.pi * system.mesh.points_m[:, 0] / plate.width_m)
            u = u.ravel()
            quotient = (u @ system.stiffness @ u) / np.dot(system.mass_kg * u, u)
            errors.append(abs(quotient / exact - 1))
        self.assertGreater(errors[0] / errors[1], 3.5)
        self.assertGreater(errors[1] / errors[2], 3.5)

    def test_degenerate_elements_bad_force_and_empty_patch_fail(self):
        with self.assertRaises(ValueError):
            element_matrices(np.zeros((1, 4, 3)), self.material)
        with self.assertRaises(ValueError):
            surface_patch(self.system, (1., 1.), 0.0001)
        with self.assertRaises(ValueError):
            simulate(self.system, 1, self.system.stable_dt_s / 2, np.zeros((1, self.system.ndof)),
                     force=lambda t: np.full(self.system.ndof, np.nan))


class PulseTests(unittest.TestCase):
    def test_burst_is_finite_and_compact(self):
        self.assertEqual(tone_burst(-1, 100000), 0)
        self.assertEqual(tone_burst(0, 100000), 0)
        self.assertEqual(tone_burst(0.001, 100000), 0)
        self.assertAlmostEqual(tone_burst(2.25e-5, 100000), (1 - np.cos(0.9 * np.pi)) / 2)
