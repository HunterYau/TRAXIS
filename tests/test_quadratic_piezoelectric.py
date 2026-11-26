import unittest
from pathlib import Path
import numpy as np
from traxis.quadratic_piezoelectric import (shape,gradients,quadrature,element_matrices,
    quadratic_disk,assemble,attach_quadratic_disk,EDGES)
from traxis.piezoelectric import PiezoMaterial,read_profile,PiezoSystem,electrode_port
from traxis.pzt_receiver import DiskGeometry
from traxis.gpu_tensor import make_layout
from traxis.config import Plate
from traxis.pzt_body import merge_attachments

PROFILE=Path(__file__).resolve().parents[1]/'configs/pzt/nce51-noliac-ver1404.json'

def material():
    return PiezoMaterial.from_profile(read_profile(next(PROFILE.parent.glob('nce51*.json'))))

class QuadraticTests(unittest.TestCase):
    def setUp(self):
        self.material=material();p=np.array([[0.,0.,0.],[.001,0,0],[0,.0007,0],[0,0,.0001]])
        self.points=np.vstack([p,[.5*(p[i]+p[j]) for i,j in EDGES]])

    def test_interpolation_partition_and_linear_quadratic_completeness(self):
        bary,w=quadrature();x=bary@self.points[:4];N=shape(bary)
        self.assertLess(abs(N.sum(1)-1).max(),1e-14)
        np.testing.assert_allclose(N@self.points,x,atol=1e-18)
        np.testing.assert_allclose(N@(self.points[:,0]**2),x[:,0]**2,atol=1e-20)
        jac=(self.points[1:4]-self.points[0]).T;g=np.empty((3,4));g[:,1:]=np.linalg.inv(jac).T;g[:,0]=-g[:,1:].sum(1)
        grad=gradients(bary,g)
        np.testing.assert_allclose(grad.sum(2),0,atol=1e-10)
        np.testing.assert_allclose(np.einsum('qji,i->qj',grad,self.points[:,0]**2),np.c_[2*x[:,0],np.zeros((len(x),2))],atol=1e-16)

    def test_exact_consistent_mass_and_independent_higher_quadrature(self):
        a=element_matrices(self.points,self.material);b=element_matrices(self.points,self.material,stiffness_order=4,mass_order=5)
        for x,y in zip(a[:4],b[:4]):self.assertLess(np.linalg.norm(x-y)/np.linalg.norm(y),1e-13)
        np.linalg.cholesky(a[1])
        trans=np.r_[np.ones(10),np.zeros(20)]
        self.assertAlmostEqual((trans@a[1]@trans)/(self.material.density_kg_m3*a[4]),1,places=13)
        with self.assertRaises(ValueError):element_matrices(self.points,self.material,mass_order=2)
        curved=self.points.copy();curved[4,2]+=.00001
        with self.assertRaises(ValueError):element_matrices(curved,self.material)

    def test_bending_energy_and_rigid_motion(self):
        K,M,B,D,v=element_matrices(self.points,self.material);p=self.points
        u=np.zeros((10,3));u[:,0]=-p[:,0]*p[:,2];u[:,2]=.5*p[:,0]**2;u=u.T.ravel()
        z=p[:4,2];exact=self.material.stress_charge[0][0,0]*v*(z.sum()**2+z@z)/20
        self.assertAlmostEqual((u@K@u)/exact,1,places=12)
        rot=np.cross([.1,.2,.3],p).T.ravel()
        self.assertLess(np.linalg.norm(K@rot)/(np.linalg.norm(K)*np.linalg.norm(rot)),1e-14)
        self.assertLess(np.linalg.norm(D@np.ones(10))/np.linalg.norm(D),1e-14)

    def test_shared_midpoints_and_uniform_electrode_capacitance(self):
        geo=DiskGeometry((.5,.5),.005,.0005,'top',1);m=quadratic_disk(geo,rings=1,sectors=8,layers=1)
        # All coincident topological nodes must be shared.
        self.assertEqual(len(m.points_m),len(np.unique(np.round(m.points_m,15),axis=0)))
        a=assemble(m,self.material);z=m.points_m[:,2]
        port=electrode_port(PiezoSystem(a.stiffness.toarray(),a.mass.toarray(),a.coupling.toarray(),a.dielectric.toarray(),a.volume_m3),np.flatnonzero(z==0),np.flatnonzero(z==.0005))
        expected=self.material.stress_charge[2][2,2]*a.volume_m3/.0005**2
        self.assertAlmostEqual(port.capacitance_f/expected,1,places=12)
        np.testing.assert_allclose(port.potential_per_v,z/.0005,atol=1e-13)

    def test_plate_attachment_includes_midpoints_and_preserves_rigid_motion(self):
        layout=make_layout(Plate(.04,.04,.001),(2,2,1),2)
        a=attach_quadratic_disk(layout,self.material,DiskGeometry((.02,.02),.003,.0005,'top',1),rings=1,sectors=8,layers=1,max_workspace_mib=256)
        v=np.zeros(len(a.indices));host=layout.screen['vector_dofs'];ns=host//3
        v[(a.indices>=0)&(a.indices<ns)]=1
        # Free body x DOFs follow the host; use the retained body DOF mapping.
        free_x=a.free_body_dofs<len(a.points_m)
        v[np.searchsorted(a.indices,host+np.flatnonzero(free_x))]=1
        expected=np.r_[np.ones(len(a.points_m)),np.zeros(2*len(a.points_m))]
        np.testing.assert_allclose(a.transform@v,expected,atol=1e-13)
        self.assertLess(np.linalg.norm(a.stiffness@v)/(np.linalg.norm(a.stiffness)*np.linalg.norm(v)),1e-13)
        self.assertAlmostEqual((v@a.mass@v)/a.report['body_mass_kg'],1,places=11)

    def test_two_sided_bodies_merge_with_existing_consistent_mass_backend(self):
        layout=make_layout(Plate(.04,.04,.001),(2,2,1),2)
        top=attach_quadratic_disk(layout,self.material,DiskGeometry((.02,.02),.003,.0005,'top',1),rings=1,sectors=8,layers=1,max_workspace_mib=256)
        bottom=attach_quadratic_disk(layout,self.material,DiskGeometry((.02,.02),.003,.0005,'bottom',-1),first_dof=top.next_dof,rings=1,sectors=8,layers=1,max_workspace_mib=256)
        diagonal,patches=merge_attachments(np.ones(layout.screen['vector_dofs']),[top,bottom],max_workspace_mib=256)
        self.assertEqual(len(diagonal),bottom.next_dof)
        self.assertTrue(all(p.inverse_error<1e-8 for p in patches))
        for p in patches:np.linalg.cholesky(p.mass)

if __name__=='__main__':unittest.main()
