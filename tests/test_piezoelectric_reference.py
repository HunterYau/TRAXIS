"""Physical invariants for the bounded coupled reference, without native imports."""

import unittest

import numpy as np

from examples.piezoelectric_reference import (assemble,checked_material,circuit_check,
    electrode_port,synthetic_material)
from traxis.config import Plate
from traxis.geometry import structured_mesh


class PiezoelectricReferenceTests(unittest.TestCase):
    def test_polarity_reversal_changes_voltage_sign_and_preserves_passive_stiffness(self):
        mesh = structured_mesh(Plate(width_m=.01,height_m=.008,thickness_m=.002),(1,1,2))
        c,e,epsilon,rho = synthetic_material()
        bottom,top = np.flatnonzero(mesh.points_m[:,2]==0),np.flatnonzero(mesh.points_m[:,2]==.002)
        ports = []
        for coupling in (e,-e,np.zeros_like(e)):
            k,m,b,d,_ = assemble(mesh,(c,coupling,epsilon,rho))
            ports.append(electrode_port(k,m,b,d,bottom,top))
        positive,negative,uncoupled = ports
        np.testing.assert_allclose(positive['k'],negative['k'],rtol=1e-14)
        np.testing.assert_allclose(positive['g'],-negative['g'],rtol=1e-14)
        self.assertAlmostEqual(positive['cp']/(epsilon[2,2]*.01*.008/.002),1.,places=13)
        np.testing.assert_array_equal(uncoupled['g'],np.zeros_like(uncoupled['g']))
        self.assertGreater(np.linalg.norm(positive['g']),0.)
        force = np.arange(1,len(positive['g'])+1,dtype=float)
        voltages = []
        for port in ports[:2]:
            koc = port['k']+np.outer(port['g'],port['g'])/port['cp']
            u = np.linalg.solve(koc,force)
            voltages.append(port['g']@u/port['cp'])
        self.assertGreater(abs(voltages[0]),1e-6)
        self.assertAlmostEqual(voltages[0]/voltages[1],-1.,places=13)

    def test_isolated_charge_and_resistive_current_power_balance(self):
        port = dict(k=np.array([[7.]]),m=np.array([[2.]]),g=np.array([.5]),cp=.3)
        isolated,_ = circuit_check(port,0.,0.,64)
        self.assertLess(isolated['isolated_charge_relative_range'],1e-12)
        self.assertLess(isolated['total_energy_balance_relative_error'],1e-12)
        passive,_ = circuit_check(port,.1,0.,64)
        self.assertLess(passive['final_energy_j'],passive['initial_energy_j'])
        self.assertGreater(passive['dissipated_energy_j'],0.)
        self.assertLess(passive['total_energy_balance_relative_error'],1e-12)
        driven,_ = circuit_check(port,.1,1e-6,64)
        self.assertNotEqual(driven['supplied_energy_j'],0.)
        self.assertLess(driven['total_energy_balance_relative_error'],1e-12)

    def test_rejects_nonpassive_material_and_overlapping_electrodes(self):
        c,e,epsilon,rho = synthetic_material()
        with self.assertRaises(ValueError):
            checked_material(c+np.triu(np.ones_like(c))*1e10,e,epsilon,rho)
        with self.assertRaises(np.linalg.LinAlgError):
            checked_material(c,e,-epsilon,rho)
        with self.assertRaises(ValueError):
            checked_material(c,e.astype(complex),epsilon,rho)
        with self.assertRaises(ValueError):
            electrode_port(np.eye(6),np.eye(6),np.ones((6,2)),np.eye(2),[0],[0])
        port = dict(k=np.eye(1),m=np.eye(1),g=np.ones(1),cp=1.)
        with self.assertRaises(ValueError):
            circuit_check(port,-.1,0.,64)


if __name__=='__main__':
    unittest.main()
