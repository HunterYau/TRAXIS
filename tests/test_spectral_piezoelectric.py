import unittest
import numpy as np
from scipy.sparse.linalg import splu
from traxis.spectral_piezoelectric import disk_mesh,assemble,element_matrices,basis
from traxis.quadratic_piezoelectric import electrical_reduction
from traxis.pzt_receiver import DiskGeometry
from tests.test_quadratic_piezoelectric import material


class SpectralTests(unittest.TestCase):
    def setUp(self):
        self.geometry=DiskGeometry((.5,.5),.005,.0005,'top',1)
        self.mesh=disk_mesh(self.geometry,side_cells=2,order=2)
        self.material=material()

    def test_topology_volume_mass_and_uniform_electrodes(self):
        m=self.mesh;a=assemble(m,self.material);n=len(m.points_m)
        self.assertLess(abs(a.volume_m3/(m.polygon_area_m2*.0005)-1),1e-13)
        _,_,bottom,top=electrical_reduction(a,m.points_m)
        self.assertEqual(len(bottom),len(top))
        B,D,_,_=electrical_reduction(a,m.points_m)
        # Uniform top voltage, relaxed interior potential, grounded bottom.
        phi=np.r_[-splu(D[:-1,:-1]).solve(D[:-1,-1].toarray().ravel()),1.]
        cap=float(phi@(D@phi));eps=self.material.stress_charge[2]
        self.assertLess(abs(cap/(eps[2,2]*m.polygon_area_m2/.0005)-1),1e-12)
        u=np.r_[m.points_m[:,0],np.zeros(2*n)]*1e-6
        charge=u@(B@phi);e=self.material.stress_charge[1]
        self.assertLess(abs(charge/(e[2,0]*1e-6*m.polygon_area_m2)-1),1e-12)
        ones=np.r_[np.ones(n),np.zeros(2*n)]
        self.assertLess(abs((ones@a.mass@ones)/(self.material.density_kg_m3*a.volume_m3)-1),1e-12)
        self.assertLess(np.linalg.norm(a.stiffness@ones)/(np.linalg.norm(a.stiffness.data)*np.linalg.norm(ones)),1e-13)

    def test_quadratic_bending_and_quadrature_refinement(self):
        # Every central and annular cell, including non-affine geometry.
        for ids in self.mesh.hexahedra:
            points=self.mesh.points_m[ids];nl=len(ids);K,M,B,D,v=element_matrices(points,self.material,2)
            k2,m2,b2,d2,_=element_matrices(points,self.material,2,6)
            # Variable metric stiffness needs overintegration; manufactured
            # physical quadratic bending is represented exactly by Q2.
            x,y,z=points.T;u=np.r_[-x*(z-.00025),np.zeros(nl),.5*x*x]
            exact=self.material.stress_charge[0][0,0]*v*.0005**2/12
            # Forming u.T K u subtracts stiff membrane terms to recover a
            # much smaller thin-disk bending energy; allow FP64 cancellation.
            self.assertLess(abs(u@K@u/exact-1),5e-10)
            self.assertLess(np.linalg.norm(M-m2)/np.linalg.norm(m2),1e-12)
            self.assertLess(np.linalg.norm(K-K.T)/np.linalg.norm(K),1e-13)
            np.linalg.cholesky(M)
        with self.assertRaises(ValueError):element_matrices(points,self.material,2,2)

    def test_q3_manufactured_electrical_and_mechanical_fields(self):
        mesh=disk_mesh(self.geometry,side_cells=2,order=3)
        a=assemble(mesh,self.material,quadrature_order=6);n=len(mesh.points_m)
        B,D,lower,upper=electrical_reduction(a,mesh.points_m)
        phi=np.r_[-splu(D[:-1,:-1]).solve(D[:-1,-1].toarray().ravel()),1.]
        cap=float(phi@(D@phi));c,e,eps=self.material.stress_charge
        self.assertLess(abs(cap/(eps[2,2]*mesh.polygon_area_m2/.0005)-1),1e-12)
        x,y,z=mesh.points_m.T;u=np.r_[-x*(z-.00025),np.zeros(n),.5*x*x]
        expected=c[0,0]*a.volume_m3*.0005**2/12
        self.assertLess(abs(u@a.stiffness@u/expected-1),1e-9)
        self.assertLess(abs(a.volume_m3/(mesh.polygon_area_m2*.0005)-1),1e-13)

    def test_shared_faces_are_conforming(self):
        # Each z-layer has the identical in-plane node set; no duplicate nodes.
        p=self.mesh.points_m
        self.assertEqual(len(np.unique(np.round(p,14),axis=0)),len(p))
        for z in np.unique(p[:,2]):
            self.assertEqual(np.count_nonzero(p[:,2]==z),len(p)//3)
        N,G,w=basis(3,5)
        np.testing.assert_allclose(N.sum(1),1,atol=1e-14)
        np.testing.assert_allclose(G.sum(2),0,atol=1e-13)


if __name__=='__main__':unittest.main()
