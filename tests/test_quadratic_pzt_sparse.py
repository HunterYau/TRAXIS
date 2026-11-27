"""Small FP64 dense-reference checks only; no plate propagation."""
import unittest
import numpy as np
from traxis.config import Plate
from traxis.gpu_tensor import make_layout
from traxis.pzt_receiver import DiskGeometry
from traxis.quadratic_piezoelectric import attach_quadratic_disk
from traxis.quadratic_pzt_sparse import component,SparseAttachment,ConsistentMass
from tests.test_quadratic_piezoelectric import material
from traxis.local_time import split_lf2_step


class SparseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout=make_layout(Plate(.04,.04,.001),(2,2,1),2)
        cls.geometry=DiskGeometry((.02,.02),.003,.0005,'top',1)
        cls.data=component(cls.geometry,material(),rings=1,sectors=8,layers=1)

    def pairs(self):
        for surface,sign in [('top',1),('bottom',-1)]:
            geo=DiskGeometry((.02,.02),.003,.0005,surface,sign)
            for electrical in ['open','voltage']:
                for cap in [0.,2e-9]:
                    dense=attach_quadratic_disk(self.layout,material(),geo,rings=1,sectors=8,layers=1,
                        electrical=electrical,input_capacitance_f=cap)
                    sparse=SparseAttachment(self.data,self.layout,geo,electrical=electrical,input_capacitance_f=cap)
                    yield dense,sparse

    def test_sparse_schedule_rejects_incompatible_execution_adapters(self):
        from examples import pzt_mesh_protocol as protocol
        # Reject before baseline reads, GPU setup or output creation.
        for options in [dict(),dict(skip_stability_checks=True),dict(execution_backend='original')]:
            with self.assertRaisesRegex(ValueError,'Sparse study requires'):
                protocol.prepare('unused-output',None,None,pzt_storage='sparse',**options)

    def test_stiffness_mass_port_energy_reciprocity_both_faces(self):
        rng=np.random.default_rng(188)
        for a,b in self.pairs():
            np.testing.assert_array_equal(a.indices,b.indices)
            self.assertLess(abs(a.capacitance_f/b.capacitance_f-1),1e-12)
            self.assertLess(np.linalg.norm(a.coupling-b.coupling)/np.linalg.norm(a.coupling),1e-12)
            x=rng.normal(size=len(a.indices));y=rng.normal(size=len(x))
            self.assertLess(np.linalg.norm(a.stiffness@x-b.stiffness_action(x))/np.linalg.norm(a.stiffness@x),1e-12)
            self.assertLess(np.linalg.norm(a.mass@x-b.body_mass_action(x))/np.linalg.norm(a.mass@x),1e-12)
            self.assertLess(abs(x@b.stiffness_action(y)-y@b.stiffness_action(x))/(np.linalg.norm(a.stiffness)*np.linalg.norm(x)*np.linalg.norm(y)),1e-13)
            self.assertGreater(x@b.stiffness_action(x),0)

    def test_consistent_mass_inverse_and_original_short_lts_recurrence(self):
        a,b=next(self.pairs());n=len(a.indices);host=np.ones(self.layout.screen['vector_dofs'])*1e-3
        m=ConsistentMass(b,host);full=a.mass.copy()
        ids=np.flatnonzero(a.indices<len(host));full[ids,ids]+=host[a.indices[ids]]
        rng=np.random.default_rng(71);x=rng.normal(size=n)
        self.assertLess(np.linalg.norm(m.action(x)-full@x)/np.linalg.norm(full@x),1e-12)
        self.assertLess(np.linalg.norm(m.solve(x)-np.linalg.solve(full,x))/np.linalg.norm(np.linalg.solve(full,x)),1e-12)
        self.assertLess(np.linalg.norm(m.action(m.solve(x))-x)/np.linalg.norm(x),1e-12)
        A=np.linalg.solve(full,a.stiffness)
        self.assertLess(np.linalg.eigvalsh(np.linalg.solve(np.linalg.cholesky(full),a.stiffness)@np.linalg.inv(np.linalg.cholesky(full).T)).max(),b.data.body_eigenvalue_bound)
        action=lambda x:m.solve(b.stiffness_action(x))
        dense=lambda x:A@x
        fine=np.ones(n,dtype=bool);h=1e-9;load=np.zeros(n)
        prev=rng.normal(size=n)*1e-10;cur=rng.normal(size=n)*1e-10
        dp,dc=prev.copy(),cur.copy()
        for j in range(16):
            nxt=split_lf2_step(prev,cur,h,2,action,action,fine,load,lambda t:0.,j*h)
            dn=split_lf2_step(dp,dc,h,2,dense,dense,fine,load,lambda t:0.,j*h)
            prev,cur=cur,nxt;dp,dc=dc,dn
        self.assertLess(np.linalg.norm(cur-dc)/np.linalg.norm(dc),1e-12)

    def test_six_frequency_voltage_response(self):
        a,b=next(self.pairs());n=len(a.indices);eye=np.eye(n)
        K=np.column_stack([b.stiffness_action(x) for x in eye]);M=np.column_stack([b.body_mass_action(x) for x in eye])
        # Clamp all host interface support exactly in both formulations.
        free=np.flatnonzero(a.indices>=self.layout.screen['vector_dofs'])
        force=np.random.default_rng(2).normal(size=len(free))
        for f in [50000,100000,150000,200000,250000,300000]:
            w2=(2*np.pi*f)**2
            x=np.linalg.solve((K-w2*M)[np.ix_(free,free)],force)
            y=np.linalg.solve((a.stiffness-w2*a.mass)[np.ix_(free,free)],force)
            va=b.coupling[free]@x/b.total_capacitance_f;vb=a.coupling[free]@y/a.total_capacitance_f
            self.assertLess(abs(va-vb)/abs(vb),1e-11)


if __name__=='__main__':unittest.main()
