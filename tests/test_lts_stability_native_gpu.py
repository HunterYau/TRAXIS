"""Native CuPy Lanczos checks on tiny operators only; no time integration."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import eigvalsh
from traxis import lts_stability as screen
from tests.test_lts_stability import TinyCudaScreenTests

NATIVE_RESULTS={}


class NativeGpuScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cupy as cp
        cls.cp=cp
        screen._gpu_available(cp)  # Missing native libraries must fail, not skip.

    def fixture(self):
        case=TinyCudaScreenTests();case.cp=self.cp
        s,f,h=case.system()
        return case,s,f,h

    def test_native_two_start_matches_dense_reference_and_original_cpu(self):
        case,s,f,h=self.fixture()
        exact=eigvalsh(case.dense(s,f,h))[-1]
        old=screen.original.lts_stability_screen(s,f,h,2)
        report=screen.optimized_stability_screen(s,f,h,2,backend='gpu')
        self.assertEqual(report['backend'],'gpu')
        self.assertFalse(report['cache_hit'])
        self.assertGreater(report['operator_evaluations'],0)
        self.assertEqual(len(report['residuals']),2)
        for value in report['estimated_H2_lambda_max']:
            self.assertAlmostEqual(value,exact,places=7)
            self.assertAlmostEqual(value,old['estimated_H2_lambda_max'][0],places=7)
        self.assertLessEqual(max(report['residuals']),1e-8)
        NATIVE_RESULTS['reference_comparison']=dict(dofs=len(f),dense_lambda_max=float(exact),
            original_cpu_estimates=old['estimated_H2_lambda_max'],
            native_gpu_estimates=report['estimated_H2_lambda_max'],native_gpu_residuals=report['residuals'],
            max_gpu_dense_absolute_error=max(abs(x-exact) for x in report['estimated_H2_lambda_max']))

    def test_native_gpu_cache_reuse_and_nearby_step_warm_start(self):
        case,s,f,h=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            first=screen.optimized_stability_screen(s,f,h,2,backend='gpu',cache_dir=tmp)
            _,fresh,ff,_=self.fixture()
            with patch.object(screen,'_gpu_available',side_effect=AssertionError('Exact cache should bypass iteration')):
                cached=screen.optimized_stability_screen(fresh,ff,h,2,backend='gpu',cache_dir=tmp)
            self.assertTrue(cached['cache_hit']);self.assertEqual(cached['operator_evaluations'],0)
            self.assertEqual(first['estimated_H2_lambda_max'],cached['estimated_H2_lambda_max'])
            changed=screen.optimized_stability_screen(fresh,ff,h*1.001,2,backend='gpu',cache_dir=tmp)
            self.assertFalse(changed['cache_hit']);self.assertTrue(changed['warm_start_used'])
            exact=eigvalsh(case.dense(fresh,ff,h*1.001))[-1]
            for value in changed['estimated_H2_lambda_max']:self.assertAlmostEqual(value,exact,places=7)
            NATIVE_RESULTS['cache_and_warm']=dict(cache_hit=cached['cache_hit'],
                cache_operator_evaluations=cached['operator_evaluations'],
                warm_start_used=changed['warm_start_used'],warm_residuals=changed['residuals'])

    def test_wrong_subdominant_warm_hint_cannot_certify_wrong_eigenvalue(self):
        case,s,f,h=self.fixture();B=case.dense(s,f,h)
        values,vectors=np.linalg.eigh(B)
        with tempfile.TemporaryDirectory() as tmp:
            screen._save_warm(Path(tmp)/'warm-latest.npz',screen.model_fingerprint(s,f),vectors[:,0])
            report=screen.optimized_stability_screen(s,f,h,2,backend='gpu',cache_dir=tmp)
        self.assertTrue(report['warm_start_used'])
        for value in report['estimated_H2_lambda_max']:self.assertAlmostEqual(value,values[-1],places=7)
        NATIVE_RESULTS['subdominant_hint_recovered_correct_maximum']=True

    def test_native_gpu_eigenvalues_over_rejection_threshold_are_rejected(self):
        _,s,f,h=self.fixture();cp=self.cp
        # Independent controlled spectral operator, passed through native CuPy
        # eigsh and the production acceptance logic (not a fabricated result).
        class UnsafeOperator:
            calls=0
            def __call__(self,z):
                self.calls+=1
                return cp.linspace(.1,3.9,len(z),dtype=cp.float64)*z
        with patch.object(screen,'NormalizedOperator',return_value=UnsafeOperator()):
            with self.assertRaisesRegex(ArithmeticError,'fails effective-operator stability screen'):
                screen.optimized_stability_screen(s,f,h,2,backend='gpu')
        NATIVE_RESULTS['unsafe_3_9_operator_rejected']=True


if __name__=='__main__':unittest.main()
