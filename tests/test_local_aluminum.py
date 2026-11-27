"""Small mesh-adapter checks; no full plate or production GPU allocations."""
import unittest
import numpy as np
from traxis.config import Material,Plate
from traxis.gpu_tensor import make_layout
from traxis.local_aluminum import LocalMesh,point_weights


class LocalAluminumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.material=Material(71e9,.33,2700)
        cls.layout=make_layout(Plate(.024,.024,.001),(6,6,1),2)
        cls.mesh=LocalMesh(cls.layout,cls.material,[(.012,.012)],patch_width_m=.012,cells_per_patch_axis=4)
    def test_outside_coordinates_and_polynomials(self):
        m=self.mesh;xyz=m.points()
        np.testing.assert_array_equal(xyz[:len(m.keep)],self.layout.points()[m.keep])
        for point in ((.002,.003,.0003),(.007,.012,.001),(.012,.013,.0002),(.020,.010,.0007)):
            ids,w=m.weights(point)
            self.assertLess(abs(w.sum()-1),2e-13)
            np.testing.assert_allclose(w@xyz[ids],point,atol=3e-17,rtol=0)
            value=xyz[:,0]**2+xyz[:,0]*xyz[:,1]+xyz[:,2]**2
            self.assertAlmostEqual(w@value[ids],point[0]**2+point[0]*point[1]+point[2]**2,places=15)
    def test_trace_continuity_between_nodes(self):
        m=self.mesh;q=m.patches[0];rng=np.random.default_rng(7);u=rng.normal(size=m.n)
        fine=q.t@u[q.support];old=np.zeros(m.old_n);old[m.keep]=u[:len(m.keep)]
        for y in np.linspace(q.bounds[1][0],q.bounds[1][1],45):
            pt=(q.bounds[0][0],y,.00037)
            a,w=point_weights(q.block.knots,m.base.order,pt);b,v=point_weights(m.coarse_knots,m.base.order,pt)
            self.assertAlmostEqual(w@fine[a],v@old[b],places=11)
    def test_stiffness_symmetry_rigid_motion(self):
        m=self.mesh;xyz=m.points();rng=np.random.default_rng(9)
        x=rng.normal(size=3*m.n);y=rng.normal(size=3*m.n)
        kx=m.stiffness(x);ky=m.stiffness(y)
        self.assertLess(abs(x@ky-y@kx)/(np.linalg.norm(x)*np.linalg.norm(ky)),2e-14)
        rotation=np.c_[-xyz[:,1],xyz[:,0],np.zeros(m.n)].T.ravel()
        self.assertLess(np.linalg.norm(m.stiffness(rotation))/(np.linalg.norm(kx)*np.linalg.norm(rotation)/np.linalg.norm(x)),1e-12)
        self.assertGreater(x@kx,0)
    def test_projected_mass_kept_exact(self):
        m=self.mesh;rng=np.random.default_rng(10);x=rng.normal(size=3*m.n)
        np.testing.assert_allclose(m.mass(m.mass(x),inverse=True),x,rtol=3e-13,atol=3e-13)
        one=np.ones(3*m.n)
        expected=3*.024*.024*.001*self.material.density_kg_m3
        self.assertAlmostEqual(one@m.mass(one),expected,places=14)
        self.assertTrue(any(b.matrix.nnz>len(b.ids) for b in m.mass_blocks))
    def test_duplicate_receiver_location_is_refined_once(self):
        m=LocalMesh(self.layout,self.material,[(.012,.012),(.012,.012)],patch_width_m=.012,cells_per_patch_axis=4)
        self.assertEqual(len(m.patches),1)
    def test_fifth_order_trace(self):
        layout=make_layout(Plate(.024,.024,.001),(6,6,1),5,max_dofs=100000)
        m=LocalMesh(layout,self.material,[(.012,.012)],patch_width_m=.012,cells_per_patch_axis=4)
        q=m.patches[0];rng=np.random.default_rng(3);u=rng.normal(size=m.n);f=q.t@u[q.support]
        old=np.zeros(m.old_n);old[m.keep]=u[:len(m.keep)]
        for y in np.linspace(q.bounds[1][0],q.bounds[1][1],35):
            pt=(q.bounds[0][0],y,.00037);a,w=point_weights(q.block.knots,5,pt);b,v=point_weights(m.coarse_knots,5,pt)
            self.assertAlmostEqual(w@f[a],v@old[b],places=10)

if __name__=='__main__':unittest.main()
