import unittest
import numpy as np
from tests.test_pzt_receiver import material
from traxis.config import Plate
from traxis.gpu_tensor import make_layout
from traxis.pzt_body import attach_disk
from traxis.pzt_receiver import DiskGeometry
from traxis.experimental_setup import applied_actuator_voltage


class ElasticBondTests(unittest.TestCase):
    def test_recorded_drive_is_amplified_and_delayed_without_peak_normalization(self):
        t=np.arange(5)*1e-6;voltage=np.array([0.,.2,.5,.1,0.])
        actual=applied_actuator_voltage(t,t,voltage,{'gain_v_v':10.,'delay_s':1e-6})
        np.testing.assert_allclose(actual,[0.,0.,2.,5.,1.],atol=1e-14)

    def test_rigid_motion_mass_and_uniform_slip_energy(self):
        layout=make_layout(Plate(.04,.04,.001),(2,2,1),2)
        profile=dict(young_pa=1e9,poisson=.3,density_kg_m3=1100.,thickness_m=1e-4)
        b=attach_disk(layout,material(),DiskGeometry((.02,.02),.005,.0005,'top',1),
            sectors=8,rings=1,layers=1,electrical='voltage',bond='elastic_interface',adhesive=profile)
        n=layout.screen['vector_dofs']
        for component in range(3):
            def field(x):
                u=np.zeros_like(x);u[:,component]=1.;return u
            full=np.r_[field(layout.points()).T.ravel(),field(b.points_m).T.ravel()]
            u=full[b.indices]
            total=b.report['body_mass_kg']+b.report['adhesive']['adhesive_mass_kg']
            self.assertAlmostEqual(float(u@b.mass@u)/total,1.,places=10)
            self.assertLess(abs(u@b.stiffness@u)/(np.linalg.norm(b.stiffness)*np.dot(u,u)),1e-13)
            slip=full.copy();slip[:n]=0.;slip*=1e-6;local=slip[b.indices]
            e=profile['young_pa'];nu=profile['poisson']
            modulus=e/(2*(1+nu)) if component<2 else e*(1-nu)/((1+nu)*(1-2*nu))
            expected=b.report['adhesive']['area_m2']*modulus/profile['thickness_m']*1e-12
            self.assertAlmostEqual(float(local@b.stiffness@local)/expected,1.,places=10)
        rotation=lambda x:np.cross(np.array([.7,-.3,.2]),x)
        u=np.r_[rotation(layout.points()).T.ravel(),rotation(b.points_m).T.ravel()][b.indices]
        self.assertLess(abs(u@b.stiffness@u)/(np.linalg.norm(b.stiffness)*np.dot(u,u)),1e-13)
        self.assertGreater(len(b.free_body_dofs),0)
