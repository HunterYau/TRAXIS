"""Screening only: dense spectral references, invalidation and fallback checks.

No time integration or full-plate simulation is executed by this suite.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import cholesky,solve_triangular,eigvalsh
from scipy.sparse.linalg import LinearOperator,eigsh
from traxis import lts_stability as screen
from traxis.config import Plate,Material
from traxis.gpu_tensor import make_layout,build_gpu_tensor
from traxis.gpu_pzt_body import couple_gpu_bodies
from traxis.local_time import fine_partition
from traxis.pzt_body import attach_disk
from traxis.pzt_receiver import DiskGeometry
from tests.test_pzt_receiver import material


class CertificateTests(unittest.TestCase):
    def test_bounds_dominate_independent_dense_effective_spectra(self):
        rng=np.random.default_rng(73)
        for n in (3,11,27):
            Q,_=np.linalg.qr(rng.standard_normal((n,n)))
            S=(Q*np.linspace(0.01,10,n))@Q.T
            dt=2/np.sqrt(10.)
            for ratio in (.81,.96,1.05,1.59):
                H=ratio*dt;T=H*H*S
                for all_fine in (False,True):
                    P=np.diag(np.ones(n) if all_fine else rng.integers(0,2,n))
                    B=T-T@P@T/16
                    self.assertGreaterEqual(eigvalsh(B)[0],-1e-12)
                    self.assertLessEqual(eigvalsh(B)[-1],screen.conservative_upper(H,dt,all_fine))

    def test_near_threshold_and_failed_residuals_never_accept(self):
        good=dict(accepted=True,estimated_H2_lambda_max=[1.9,1.9],residuals=[1e-10,1e-10])
        self.assertTrue(screen._valid_report(good))
        for edit in (dict(estimated_H2_lambda_max=[3.8,3.8]),
                     dict(residuals=[1e-5,1e-10]),
                     dict(estimated_H2_lambda_max=[1.9,1.91]),
                     dict(estimated_H2_lambda_max=[float('nan'),1.9])):
            self.assertFalse(screen._valid_report(dict(good,**edit)))

    def test_corrupt_disk_certificate_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.json';p.write_text('{')
            self.assertIsNone(screen._read_report(p,'expected'))
            p.write_text(json.dumps(dict(key='wrong',criteria=screen.CRITERIA,report={})))
            self.assertIsNone(screen._read_report(p,'expected'))


class TinyCudaScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount()<1: raise unittest.SkipTest('No CUDA GPU')
        cls.cp=cp

    def system(self):
        layout=make_layout(Plate(.04,.04,.001),(1,1,1),1)
        base=build_gpu_tensor(layout,Material())
        body=attach_disk(layout,material(),DiskGeometry((.02,.02),.005,.0005,'top',1),
            sectors=8,rings=1,layers=1,electrical='open')
        system=couple_gpu_bodies(base,[body]);n=len(system.mass_diagonal)
        load=np.zeros(n);load[-1]=1
        fine=fine_partition(system,load)
        return system,fine,1.05*system.stable_dt_s

    def dense(self,system,fine,H):
        cp=self.cp;n=len(fine);actions=screen.original.LocalActions(system,fine)
        A=np.column_stack([cp.asnumpy(actions.full(cp.asarray(np.eye(n)[:,j]))) for j in range(n)])
        M=np.diag(system.mass_diagonal)
        for b in system.body_patches:M[np.ix_(b.indices,b.indices)]=b.mass
        L=cholesky(M,lower=True)
        return H*H*L.T@(A-H*H/16*(A*fine[None,:])@A)@solve_triangular(L.T,np.eye(n),lower=False)

    def test_cpu_screen_and_device_normalization_match_dense_mass_reference(self):
        system,fine,H=self.system();B=self.dense(system,fine,H)
        expected=eigvalsh(B)[-1]
        historical=screen.original.lts_stability_screen(system,fine,H,2)
        for value in historical['estimated_H2_lambda_max']:self.assertAlmostEqual(value,expected,places=7)
        for backend in ('cpu','gpu'):
            action=screen.NormalizedOperator(system,fine,H,backend,None)
            x=np.random.default_rng(12).standard_normal(len(fine))
            actual=action(self.cp.asarray(x) if backend=='gpu' else x)
            if backend=='gpu':actual=self.cp.asnumpy(actual)
            np.testing.assert_allclose(actual,B@x,rtol=1e-10,atol=1e-10)
        report=screen.optimized_stability_screen(system,fine,H,2,backend='cpu')
        for value in report['estimated_H2_lambda_max']:self.assertAlmostEqual(value,expected,places=7)

    def test_exact_disk_hit_bypasses_eigensolver_and_primes_original_cache(self):
        system,fine,H=self.system()
        with tempfile.TemporaryDirectory() as tmp:
            one=screen.optimized_stability_screen(system,fine,H,2,backend='cpu',cache_dir=tmp)
            fresh,fresh_fine,_=self.system()
            with patch.object(screen,'NormalizedOperator',side_effect=AssertionError('Redundant eigenvalue solve')):
                two=screen.optimized_stability_screen(fresh,fresh_fine,H,2,cache_dir=tmp)
                legacy=screen.original.lts_stability_screen(fresh,fresh_fine,H,2)
            self.assertTrue(two['cache_hit']);self.assertEqual(two['operator_evaluations'],0)
            self.assertEqual(legacy['cache_key'],two['cache_key'])
            self.assertEqual(one['estimated_H2_lambda_max'],two['estimated_H2_lambda_max'])
            two['residuals'][0]=999
            again=screen.optimized_stability_screen(fresh,fresh_fine,H,2,cache_dir=tmp)
            self.assertLess(again['residuals'][0],1e-6)

    def test_mutated_mass_stiffness_partition_and_step_cannot_reuse_certificate(self):
        system,fine,H=self.system();key=screen.model_fingerprint(system,fine)
        system.mass_diagonal[0]*=1.001
        self.assertNotEqual(key,screen.model_fingerprint(system,fine))
        system.mass_diagonal[0]/=1.001
        key=screen.model_fingerprint(system,fine)
        changed=fine.copy();changed[0]=~changed[0]
        self.assertNotEqual(key,screen.model_fingerprint(system,changed))
        storage=system.stiffness.storage;storage.patches[0][1][0,0]*=1.001
        self.assertNotEqual(key,screen.model_fingerprint(system,fine))
        with tempfile.TemporaryDirectory() as tmp:
            s,f,h=self.system()
            r=screen.optimized_stability_screen(s,f,h,2,backend='cpu',cache_dir=tmp)
            r2=screen.optimized_stability_screen(s,f,h*1.001,2,backend='cpu',cache_dir=tmp)
            self.assertFalse(r2['cache_hit']);self.assertNotEqual(r['cache_key'],r2['cache_key'])
            self.assertTrue(r2['warm_start_used'])
            self.assertEqual(len(r2['residuals']),2)

    def test_unavailable_gpu_and_insufficient_vram_use_cpu_without_relaxation(self):
        s,f,h=self.system()
        with patch.object(screen,'_gpu_available',side_effect=ImportError('optional cuBLAS missing')):
            r=screen.optimized_stability_screen(s,f,h,2)
        self.assertEqual(r['backend'],'cpu');self.assertIn('ImportError',r['fallback_reason'])
        s,f,h=self.system();s.screen=dict(s.screen,max_device_mib=.001)
        with patch.object(screen,'_gpu_available',side_effect=AssertionError('No room for GPU basis')):
            r=screen.optimized_stability_screen(s,f,h,2)
        self.assertEqual(r['backend'],'cpu')

    def test_original_backend_delegates_and_boundary_mass_split_rejects(self):
        s,f,h=self.system()
        with patch.object(screen.original,'lts_stability_screen',return_value={'original':True}) as original:
            self.assertEqual(screen.optimized_stability_screen(s,f,h,2,backend='original'),{'original':True})
            original.assert_called_once()
        f[s.body_patches[0].indices[0]]=False
        with self.assertRaises(ValueError):screen.optimized_stability_screen(s,f,h,2)

    def test_certificate_shortcut_and_missing_warm_hint_do_not_skip_required_checks(self):
        s,f,h=self.system()
        with patch.object(screen,'NormalizedOperator',side_effect=AssertionError('Analytic bound suffices')):
            r=screen.optimized_stability_screen(s,f,.9*s.stable_dt_s,2)
        self.assertTrue(r['formal_upper_bound']);self.assertEqual(r['operator_evaluations'],0)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'warm-latest.npz').write_bytes(b'broken optional hint')
            r=screen.optimized_stability_screen(s,f,h,2,backend='cpu',cache_dir=tmp)
            self.assertFalse(r['warm_start_used']);self.assertEqual(len(r['residuals']),2)

    def test_gpu_maxiter_approximation_is_rejected_even_inside_looser_residual_cap(self):
        cp=self.cp;s,f,h=self.system();B=self.dense(s,f,h)
        values,vectors=np.linalg.eigh(B)
        v=vectors[:,-1]+1e-7*vectors[:,0];v/=np.linalg.norm(v)
        # Passes the old final 1e-6 residual cap but misses eigsh tol=1e-8.
        error=np.linalg.norm(B@v-values[-1]*v)
        self.assertGreater(error,1e-8);self.assertLess(error,1e-6)
        def bad_driver(op,**kwargs):return cp.asarray(values[-1:]),cp.asarray(v[:,None])
        def adapter(shape,matvec,dtype):return None
        with patch.object(screen,'_gpu_available',return_value=(bad_driver,adapter)):
            with self.assertRaisesRegex(ArithmeticError,'requested tolerance'):
                screen.optimized_stability_screen(s,f,h,2,backend='gpu')

    def test_gpu_adapter_eigensolve_with_cpu_driver(self):
        # This exercises real GPU actions and normalization without optional
        # cuBLAS DLLs. It does NOT claim to test native CuPy Lanczos itself.
        cp=self.cp;s,f,h=self.system();expected=eigvalsh(self.dense(s,f,h))[-1]
        def adapter(shape,matvec,dtype):
            return LinearOperator(shape,matvec=lambda x:cp.asnumpy(matvec(cp.asarray(x))),dtype=dtype)
        def driver(op,**kwargs):
            kwargs['v0']=cp.asnumpy(kwargs['v0'])
            vals,vecs=eigsh(op,**kwargs)
            return cp.asarray(vals),cp.asarray(vecs)
        with patch.object(screen,'_gpu_available',return_value=(driver,adapter)):
            r=screen.optimized_stability_screen(s,f,h,2,backend='gpu')
        self.assertEqual(r['backend'],'gpu')
        for value in r['estimated_H2_lambda_max']:self.assertAlmostEqual(value,expected,places=7)


if __name__=='__main__':unittest.main()
