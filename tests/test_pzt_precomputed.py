"""Independent tiny dense condensation/mass references for spectral export."""
import unittest
import numpy as np
from scipy import sparse
from traxis.config import Plate, Material
from traxis.gpu_tensor import make_layout, mass_diagonal
from traxis.pzt_receiver import DiskGeometry
from traxis.spectral_pzt_sparse import component
from traxis.quadratic_pzt_sparse import SparseAttachment, ConsistentMass
from traxis.pzt_precomputed import matrices
from tests.test_quadratic_piezoelectric import material


class PrecomputedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout=make_layout(Plate(.04,.04,.001),(2,2,1),2)
        cls.host=mass_diagonal(cls.layout,Material())
        cls.data=component(DiskGeometry((.02,.02),.003,.0005,'top',1),material(),
            rings=1,sectors=8,layers=1)

    def test_both_faces_ports_and_nondivisible_batches(self):
        d=self.data
        # Independent dense dielectric solve before interface projection.
        k=d.stiffness.toarray()+d.bi.toarray()@np.linalg.solve(d.di.toarray(),d.bi.T.toarray())
        body_mass=sparse.block_diag([d.scalar_mass]*3).toarray()
        for surface in ('top','bottom'):
            for electrical in ('voltage','open'):
                with self.subTest(surface=surface,electrical=electrical):
                    a=SparseAttachment(d,self.layout,DiskGeometry((.02,.02),.003,.0005,surface,-1),
                        electrical=electrical,input_capacitance_f=.3*d.capacitance_f)
                    mass=ConsistentMass(a,self.host)
                    p=matrices(a,mass,block_columns=13,max_workspace_mib=64)
                    T=np.zeros((3*d.n,len(a.indices)))
                    for c in range(3):T[c*d.n:(c+1)*d.n,a.positions[:,c]]=a.t.toarray()
                    expected_k=T.T@k@T
                    if electrical=='open':expected_k+=np.outer(a.coupling,a.coupling)/a.total_capacitance_f
                    host=np.r_[self.host,np.zeros(a.next_dof-len(self.host))]
                    expected_m=T.T@body_mass@T+np.diag(host[a.indices])
                    inv=np.linalg.solve(expected_m,np.eye(len(a.indices)))
                    for actual,reference in ((p.stiffness,expected_k),(p.mass,expected_m),(p.inverse,inv)):
                        self.assertEqual(actual.dtype,np.dtype('float64'))
                        self.assertTrue(actual.flags.c_contiguous)
                        self.assertLess(np.linalg.norm(actual-reference)/np.linalg.norm(reference),1e-11)
                    self.assertLess(max(p.verification.values()),1e-10)

    def test_memory_guard_runs_before_dense_allocation(self):
        a=SparseAttachment(self.data,self.layout,DiskGeometry((.02,.02),.003,.0005,'top',1))
        mass=ConsistentMass(a,self.host)
        with self.assertRaises(MemoryError):matrices(a,mass,max_workspace_mib=.00001)


if __name__=='__main__':unittest.main()
