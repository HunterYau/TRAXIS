"""Mapped action, work, rigid fields and an independent spectral bound."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy.linalg import eigh

from examples.recess_geometry_reference import circle_reference, native_geometry
from traxis.config import CircularRecess, Material, Plate
from traxis.mapped_tensor import action, assemble_bounded, build_system, local_action
from traxis.recess_hex import make_recess_mesh
from traxis.tensor_hex import element_matrices, line_rule


class MappedTensorTests(unittest.TestCase):
    plate = Plate(.04, .03, .003)
    recess = CircularRecess(.021, .014, .008, .0015)

    def test_affine_limit_matches_independent_full_element_matrices(self):
        rng = np.random.default_rng(312)
        material = Material()
        jac = np.array([[1., .13, .04], [.03, 1., .07], [.02, .01, 1.]]) @ np.diag([.02, .015, .0007])
        for p in range(1, 6):
            n = (p+1)**3; weights = line_rule(p)[1]
            volume = np.kron(np.kron(weights, weights), weights)*np.linalg.det(jac)
            inverse = np.broadcast_to(np.linalg.inv(jac), (n, 3, 3))
            matrix = element_matrices(p, jac, material, 'gll')[0]
            states = rng.normal(size=(3, 3*n))
            measured = local_action(states, p, inverse, volume, material)
            expected = states @ matrix.T
            self.assertLess(np.linalg.norm(measured-expected)/np.linalg.norm(expected), 1e-12)
            np.testing.assert_allclose(local_action(states[0], p, inverse, volume, material), measured[0], rtol=1e-13, atol=1e-5)

    def test_connected_action_and_work_match_bounded_assembly(self):
        mesh = make_recess_mesh(self.plate, self.recess, 3)
        system = build_system(mesh, Material())
        matrix = assemble_bounded(system)
        rng = np.random.default_rng(19); fields = rng.normal(size=(2, len(system.mass_kg)))
        measured = action(system, fields)
        self.assertLess(np.linalg.norm(measured-fields@matrix.T)/np.linalg.norm(measured), 1e-12)
        self.assertLess(abs(fields[1]@measured[0]-fields[0]@measured[1])/abs(fields[0]@measured[0]), 1e-12)
        self.assertGreater(float(fields[0]@measured[0]), 0)
        self.assertFalse(system.inverse_jacobians.flags.writeable)
        self.assertFalse(system.mass_kg.flags.writeable)

    def test_physical_affine_strain_and_six_rigid_fields(self):
        material = Material(); lam, mu = material.lame
        gradient = np.array([[.13, -.07, .02], [.04, .09, -.03], [-.05, .08, .11]])
        strain = (gradient+gradient.T)/2
        circle_area = circle_reference(.008, 3, 4)[0]
        for defective in (False, True):
            mesh = make_recess_mesh(self.plate, self.recess, 3, angular_cells=4, defective=defective)
            system = build_system(mesh, material)
            volume = .04*.03*.003-(circle_area*.0015 if defective else 0.)
            self.assertLess(abs(system.mass_kg[:len(mesh.points_m)].sum()/(material.density_kg_m3*volume)-1), 1e-12)
            affine = (mesh.points_m@gradient.T).T.ravel()
            expected = volume*(lam*np.trace(strain)**2+2*mu*np.sum(strain*strain))
            self.assertLess(abs(affine@action(system, affine)/expected-1), 1e-11)
            fields = []
            for axis in np.eye(3):
                fields.append(np.broadcast_to(axis, mesh.points_m.shape).T.ravel())
                fields.append(np.cross(axis, mesh.points_m-mesh.points_m.mean(axis=0)).T.ravel())
            fields = np.asarray(fields)
            residual = action(system, fields)
            scale = np.linalg.norm(system.stiffness_majorant_kg_s2)*np.linalg.norm(fields)
            self.assertLess(np.linalg.norm(residual)/scale, 1e-12)

    def test_diagonal_energy_majorant_bounds_spectrum_including_negative_lambda(self):
        mesh = make_recess_mesh(self.plate, self.recess, 3)
        for material in (Material(), Material(poisson=-.2)):
            system = build_system(mesh, material)
            matrix = assemble_bounded(system)
            majorant = np.diag(system.stiffness_majorant_kg_s2)-matrix
            smallest = eigh(majorant, eigvals_only=True, subset_by_index=(0, 0))[0]
            self.assertGreater(smallest/np.max(system.stiffness_majorant_kg_s2), -1e-12)
            mass_root = np.sqrt(system.mass_kg)
            scaled = matrix/mass_root[:, None]/mass_root[None, :]
            largest = eigh(scaled, eigvals_only=True, subset_by_index=(len(matrix)-1, len(matrix)-1))[0]
            self.assertLessEqual(largest, system.eigenvalue_upper_s2*(1+1e-12))
            self.assertLessEqual(system.stable_dt_s, 2/np.sqrt(largest))

    def test_invalid_states_geometry_degree_and_allocation_reject(self):
        mesh = make_recess_mesh(self.plate, self.recess, 3)
        with patch('numpy.empty', side_effect=AssertionError('allocation attempted')):
            with self.assertRaises(MemoryError):
                build_system(mesh, Material(), max_bytes=1)
        with self.assertRaises(ValueError):
            build_system(make_recess_mesh(self.plate, self.recess, 2, geometry_order=3), Material())
        system = build_system(mesh, Material()); n = len(system.mass_kg)
        for values in (np.zeros(n-1), np.full(n, np.nan), np.zeros(n, dtype=complex), np.zeros((33, n))):
            with self.assertRaises(ValueError):
                action(system, values)
        with self.assertRaises(MemoryError):
            assemble_bounded(build_system(make_recess_mesh(self.plate, self.recess, 3, angular_cells=4), Material()))

    def test_native_geometry_transform_screen_precedes_native_import(self):
        mesh = SimpleNamespace(geometry_points_m=np.empty((3501, 3)), element_nodes=np.empty((1, 8)))
        with patch('builtins.__import__', side_effect=AssertionError('native import attempted')):
            with self.assertRaises(MemoryError):
                native_geometry(mesh, None, return_model=True)


if __name__ == '__main__':
    unittest.main()
