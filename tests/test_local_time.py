"""Small independent convergence, energy, forcing and interface checks."""
import unittest
from types import SimpleNamespace
import numpy as np
from traxis.local_time import lf2_step,polyphase_reconstruct,fine_partition
from traxis.gpu_local_time import selected_elements
from traxis.gpu_tensor import make_layout
from traxis.config import Plate


def trajectory(A,fine,H,p,steps,source,load):
    prev=np.zeros(len(load));u=prev.copy();out=[u.copy()]
    for n in range(steps):
        v=lf2_step(prev,u,H,p,lambda x:A@x,lambda x:A@(x*fine),fine,load,source,n*H)
        prev,u=u,v;out.append(u.copy())
    return np.array(out)


class LocalTimeTests(unittest.TestCase):
    def test_ratio_one_is_leapfrog_with_interface_coupling(self):
        A=np.array([[3.,-1.],[-1.,8.]])
        u=np.array([.2,.3]);prev=np.array([.1,.4]);fine=np.array([False,True])
        y=lf2_step(prev,u,.1,1,lambda x:A@x,lambda x:A@(x*fine),fine,np.zeros(2),lambda _:0.,0.)
        np.testing.assert_allclose(y,2*u-prev-.01*A@u,atol=1e-15)

    def test_second_order_against_independent_modal_solution(self):
        A=np.array([[4.,-2.],[-2.,70.]])
        lam,V=np.linalg.eigh(A);initial=np.array([1.,.1]);fine=np.array([False,True])
        def exact(t):return V@(np.cos(np.sqrt(lam)*t)*(V.T@initial))
        errors=[]
        for H in (.02,.01,.005):
            prev,u=exact(-H),initial.copy()
            for n in range(round(1/H)):
                nxt=lf2_step(prev,u,H,2,lambda x:A@x,lambda x:A@(x*fine),fine,np.zeros(2),lambda _:0.,n*H)
                prev,u=u,nxt
            errors.append(np.linalg.norm(u-exact(1.)))
        self.assertGreater(errors[0]/errors[1],3.8)
        self.assertGreater(errors[1]/errors[2],3.8)

    def test_consistent_mass_symmetry_and_discrete_energy(self):
        M=np.array([[2.,0,0],[0,2.,.3],[0,.3,1.]])
        K=np.array([[4.,-1.,0],[-1.,5.,-1.],[0,-1.,20.]])
        A=np.linalg.solve(M,K);fine=np.array([False,True,True]);H=.02
        def step(prev,u):return lf2_step(prev,u,H,3,lambda x:A@x,lambda x:A@(x*fine),fine,np.zeros(3),lambda _:0.,0.)
        effective=np.column_stack([(2*np.eye(3)[:,j]-step(np.zeros(3),np.eye(3)[:,j]))/H**2 for j in range(3)])
        np.testing.assert_allclose(M@effective,(M@effective).T,atol=2e-11)
        prev,u=np.array([.1,.2,.3]),np.array([.11,.19,.31]);energies=[]
        for _ in range(1000):
            d=(u-prev)/H
            energies.append(.5*d@M@d+.5*prev@M@effective@u)
            prev,u=u,step(prev,u)
        self.assertLess(np.ptp(energies)/abs(energies[0]),1e-10)

    def test_multiphase_broadband_equals_direct_without_alignment(self):
        A=np.array([[3.,-1.],[-1.,20.]])
        fine=np.array([False,True]);load=np.array([0.,1.]);H=.03;steps=180
        for p in (1,2,4):
            samples=np.sin(np.arange(steps*p+1)*H/p*8)**3
            samples[0]=0
            def source(t):
                j=round(t/(H/p));return samples[j] if 0<=j<len(samples) else 0.
            direct=trajectory(A,fine,H,p,steps,source,load)
            kernels=np.stack([trajectory(A,fine,H,p,steps,
                lambda t,j=r+1:float(abs(t-j*H/p)<H/p*.25),load) for r in range(p)])
            np.testing.assert_allclose(polyphase_reconstruct(kernels,samples),direct,rtol=1e-10,atol=1e-12)

    def test_patch_mass_closure_and_boundary_selection(self):
        layout=make_layout(Plate(.04,.03,.003),(3,2,1),2)
        n=layout.screen['vector_dofs']+2
        patch=SimpleNamespace(indices=np.array([0,3,n-2,n-1]))
        system=SimpleNamespace(layout=layout,mass_diagonal=np.ones(n),body_patches=[patch])
        load=np.zeros(n);load[-1]=1
        fine=fine_partition(system,load,edge_width_m=.001)
        self.assertTrue(fine[patch.indices].all())
        expected=[];ns=layout.screen['scalar_dofs'];support=fine[:3*ns].reshape(3,ns).any(axis=0)
        for e in range(layout.screen['elements']):
            if support[layout.element_nodes(e,e+1)].any():expected.append(e)
        np.testing.assert_array_equal(selected_elements(layout,fine),expected)

    def test_invalid_polyphase_and_steps(self):
        with self.assertRaises(ValueError):polyphase_reconstruct(np.ones((2,3,1)),np.zeros(5))
        with self.assertRaises(ValueError):lf2_step(None,None,.1,0,None,None,None,None,None,0)


if __name__=='__main__':unittest.main()
