import unittest

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import aslinearoperator

from traxis.implicit_fem import implicit_simulate
from traxis.iterative_fem import IterativeSettings, iterative_simulate


class IterativeDynamicsTests(unittest.TestCase):
    def test_coupled_consistent_mass_agrees_with_direct_and_independent_ode(self):
        mass = np.array([[2.,.3],[.3,1.]])
        stiffness = np.array([[8.,-2.],[-2.,3.]])
        initial_u, initial_v = np.array([.1,-.2]), np.array([.03,.04])
        load = lambda t:np.array([np.sin(3*t), .2*np.cos(2*t)])
        probes = np.array([[1.,.5],[-.3,1.]])
        errors = []
        for steps in (40,80,160):
            dt = .4/steps
            stride = steps//20
            result = iterative_simulate(aslinearoperator(stiffness), aslinearoperator(mass), mass.diagonal(),
                steps,dt,probes,force=load,initial_displacement=initial_u,initial_velocity=initial_v,
                output_stride=stride,stiffness_diagonal=stiffness.diagonal())
            direct = implicit_simulate(stiffness,mass,steps,dt,probes,force=load,
                initial_displacement=initial_u,initial_velocity=initial_v,output_stride=stride)
            np.testing.assert_allclose(result.displacement_m,direct.displacement_m,rtol=1e-11,atol=1e-13)
            np.testing.assert_allclose(result.energy_j,direct.energy_j,rtol=1e-11,atol=1e-13)
            np.testing.assert_array_equal(result.time_s,direct.time_s)
            inverse = np.linalg.inv(mass)
            reference = solve_ivp(lambda t,y:np.r_[y[2:],inverse@(load(t)-stiffness@y[:2])],
                (0,.4),np.r_[initial_u,initial_v],method='DOP853',rtol=1e-12,atol=1e-14,dense_output=True)
            truth = probes@reference.sol(result.time_s)[:2]
            errors.append(np.linalg.norm(result.displacement_m.T-truth))
            self.assertLess(result.maximum_relative_residual,1e-9)
        self.assertTrue(all(3.9 < a/b < 4.1 for a,b in zip(errors,errors[1:])))
        np.testing.assert_array_equal(initial_u,[.1,-.2])
        np.testing.assert_array_equal(initial_v,[.03,.04])

    def test_unforced_rigid_and_elastic_motion_conserve_energy(self):
        mass = np.array([[2.,.4],[.4,1.]])
        stiffness = np.array([[3.,-3.],[-3.,3.]])
        result = iterative_simulate(stiffness,mass,mass.diagonal(),400,.002,np.eye(2),
            initial_displacement=np.array([.1,-.1]),initial_velocity=np.array([.2,.2]))
        self.assertLess(np.ptp(result.energy_j)/abs(result.energy_j.mean()),1e-10)
        self.assertGreater(np.linalg.norm(result.final_displacement_m),0.)

    def test_nonconvergence_and_false_cg_success_cannot_return_traces(self):
        mass, stiffness = np.eye(3), np.array([[6.,-2.,0.],[-2.,5.,-1.],[0.,-1.,2.]])
        kwargs = {'force':lambda t:np.array([1.,2.,3.])}
        with self.assertRaisesRegex(ArithmeticError,'CG solve failed'):
            iterative_simulate(stiffness,mass,mass.diagonal(),2,.1,np.eye(3),
                settings=IterativeSettings(max_iterations=1),**kwargs)
        from unittest.mock import patch
        with patch('traxis.iterative_fem.cg', return_value=(np.zeros(3),0)):
            with self.assertRaisesRegex(ArithmeticError,'relative residual'):
                iterative_simulate(stiffness,mass,mass.diagonal(),2,.1,np.eye(3),**kwargs)

    def test_bounds_and_complex_inputs_are_rejected(self):
        k, m, probes = np.eye(2), np.eye(2), np.eye(2)
        for change in ({'relative_tolerance':.01},{'max_iterations':0},{'max_workspace_mib':float('nan')}):
            with self.assertRaises(ValueError):
                IterativeSettings(**change)
        with self.assertRaises(MemoryError):
            iterative_simulate(k,m,m.diagonal(),2,.1,probes,settings=IterativeSettings(max_workspace_mib=1e-6))
        for diagonal in (np.array([1.,0.]),np.array([1.,complex(1)])):
            with self.assertRaises(ValueError):
                iterative_simulate(k,m,diagonal,2,.1,probes)
        with self.assertRaises(ValueError):
            iterative_simulate(k.astype(complex),m,m.diagonal(),2,.1,probes)
        with self.assertRaises(ValueError):
            iterative_simulate(k,m,m.diagonal(),2,.1,probes,initial_displacement=np.array([1j,0]))
        with self.assertRaises(ValueError):
            iterative_simulate(k,m,m.diagonal(),2,.1,probes,stiffness_diagonal=np.array([-1.,0.]))
