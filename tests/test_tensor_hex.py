"""Independent invariants for tensor quadrature and elastic contractions."""
import unittest
from unittest.mock import patch
import numpy as np
from traxis.config import Material,Plate
from traxis.tensor_hex import (line_rule,make_grid,assemble_tensor,element_matrices,local_gll_action)
from examples.tensor_reference import parity_projection


class TensorHexTests(unittest.TestCase):
    def test_gll_moments_and_integration_by_parts(self):
        for order in range(1,5):
            nodes,weights,d,m,k,c = line_rule(order)
            self.assertTrue(np.all(weights>0))
            for degree in range(2*order):
                exact = 0. if degree%2 else 2/(degree+1)
                self.assertAlmostEqual(float(weights@nodes**degree),exact,places=12)
            boundary = np.diag([-1.]+[0.]*(order-1)+[1.])
            np.testing.assert_allclose(weights[:,None]*d+d.T*weights[None,:],boundary,atol=3e-13)
            np.testing.assert_allclose(d.T@np.diag(weights)@d,k,atol=3e-13)
            np.testing.assert_allclose(d.T@np.diag(weights),c,atol=3e-13)
            self.assertGreater(float(np.linalg.eigvalsh(m).min()),0.)
            self.assertGreater(abs(float(weights@nodes**(2*order))-2/(2*order+1)),1e-6)

    def test_tensor_stress_action_matches_full_gll_operator_on_skewed_cells(self):
        random = np.random.default_rng(419)
        material = Material()
        for order in range(1,5):
            for transform in (np.eye(3),np.array([[1,.13,.07],[.04,1,.11],[.02,.03,1.]])):
                jacobian = transform@np.diag([.02,.015,.0015])
                k,_,_ = element_matrices(order,jacobian,material,'gll')
                for _ in range(2):
                    state = random.normal(size=(3,)+(order+1,)*3)
                    result = local_gll_action(state,order,jacobian,material).ravel()
                    expected = k@state.ravel()
                    self.assertLess(np.linalg.norm(result-expected)/np.linalg.norm(expected),1e-12)

    def test_affine_strain_energy_and_six_rigid_motions(self):
        material = Material(); lam,mu = material.lame
        gradient = np.array([[.2,-.3,.4],[.1,-.5,.7],[.2,.1,.3]])
        for order in (1,2,3,4):
            grid = make_grid(Plate(width_m=.04,height_m=.03,thickness_m=.003),(1,1,1),order)
            volume = .04*.03*.003
            strain = (gradient+gradient.T)/2
            energy = volume*(lam*np.trace(strain)**2+2*mu*np.sum(strain*strain))
            state = (grid.points_m@gradient.T).T.ravel()
            for rule in ('exact','gll'):
                system = assemble_tensor(grid,material,rule)
                k = system.stiffness.toarray()
                self.assertLess(abs((state@k@state)/energy-1),1e-11)
                eigenvalues = np.linalg.eigvalsh(k)
                self.assertLess(max(abs(eigenvalues[:6]))/eigenvalues[-1],1e-12)
                self.assertGreater(eigenvalues[6]/eigenvalues[-1],1e-10)

    def test_shared_grid_mass_and_declared_quadrature_difference(self):
        material = Material(); plate = Plate(width_m=.04,height_m=.03,thickness_m=.003)
        grid = make_grid(plate,(2,1,2),3)
        gll,exact = (assemble_tensor(grid,material,r) for r in ('gll','exact'))
        scalar_mass = gll.mass_kg[:len(grid.points_m)]
        total = material.density_kg_m3*.04*.03*.003
        self.assertAlmostEqual(scalar_mass.sum()/total,1.,places=13)
        for axis,length in enumerate((.04,.03,.003)):
            for degree in range(6):
                self.assertAlmostEqual(float(scalar_mass@(grid.points_m[:,axis]/length)**degree)/total,1/(degree+1),places=12)
        np.testing.assert_array_equal(gll.mass_kg,exact.mass_kg)
        self.assertGreater(np.linalg.norm((gll.stiffness-exact.stiffness).toarray())/np.linalg.norm(exact.stiffness.toarray()),.01)

    def test_limits_and_invalid_geometry_reject_before_grid_allocation(self):
        with patch('numpy.meshgrid',side_effect=AssertionError('allocation attempted')):
            with self.assertRaises(MemoryError):
                make_grid(Plate(),(16,16,16),4)
        for order in (True,0,6,2.5):
            with self.assertRaises(ValueError):
                line_rule(order)
        with self.assertRaises(ValueError):
            element_matrices(3,np.diag([1.,1.,-1.]),Material())

    def test_lamb_projection_enforces_face_phase_and_physical_midplane_parity(self):
        grid = make_grid(Plate(width_m=.005,height_m=.005,thickness_m=.003),(1,1,2),3)
        wave = 400.
        random = np.random.default_rng(213)
        projections = []
        for mode,sign in (('A0',-1),('S0',1)):
            projection = parity_projection(grid,wave,mode)
            fields = (projection@random.normal(size=projection.shape[1])).reshape(3,4,4,7)
            np.testing.assert_allclose(fields[0,:,:,::-1],sign*fields[0],atol=1e-14)
            np.testing.assert_allclose(fields[2,:,:,::-1],-sign*fields[2],atol=1e-14)
            np.testing.assert_array_equal(fields[1],np.zeros_like(fields[1]))
            np.testing.assert_allclose(fields[:,-1],np.exp(1j*wave*.005)*fields[:,0],atol=1e-14)
            np.testing.assert_allclose(fields[:,:,-1],fields[:,:,0],atol=1e-14)
            projections.append(projection)
        system = assemble_tensor(grid,Material())
        cross = (projections[0].conj().T@system.stiffness@projections[1]).toarray()
        self.assertLess(np.linalg.norm(cross)/np.linalg.norm(system.stiffness.toarray()),1e-13)


if __name__=='__main__':
    unittest.main()
