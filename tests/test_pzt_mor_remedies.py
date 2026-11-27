import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import dlsim

from examples import pzt_mor_remedies as study
from examples import pzt_mechanical_mor as mechanical
from examples.pzt_era_delay import split_record, fit_delayed, delayed_impulse
from examples.pzt_broadband import reconstruct
from tests.test_pzt_broadband import simulate
from examples.pzt_broadband import interior_impulse
from traxis.storage import write_json, sha256_file


class DelayRemedyTests(unittest.TestCase):
    def test_nonzero_precursor_is_retained_and_delay_is_restored(self):
        t = np.arange(129) * .001
        system = (np.array([[.97]]), np.ones((1, 1)), np.array([[1.], [.9]]), np.zeros((2, 1)), .001)
        u = np.zeros(len(t)); u[1] = 1.
        _, original, _ = dlsim(system, u)
        d = 17
        h = np.vstack([np.zeros((d, 2)), original[:-d]])
        # An actual, nonzero early FIR contribution must survive separation.
        h[5] = [1e-6, -2e-6]
        model, info = fit_delayed(t, h, 1, backend='numpy', delay_samples=d)
        result, _ = delayed_impulse(model, 1, len(t))
        np.testing.assert_array_equal(result[2:2+d], h[2:2+d])
        np.testing.assert_allclose(result, h, atol=1e-11)
        self.assertTrue(info['delay_handling']['early_prefix_retained_exactly'])
        drive = np.sin(300*t); drive[0] = 0.
        np.testing.assert_allclose(reconstruct(result, drive), reconstruct(h, drive), atol=1e-10)
        self.assertEqual(info['delay_handling']['dynamic_samples'], len(t)-d)

    def test_common_automatic_prefix_budget_includes_cancelling_channels(self):
        t = np.arange(129)*.001
        h = np.zeros((129,2)); h[30:,0] = .98**np.arange(99); h[30:,1] = h[30:,0]
        h[4] = [1e-8, -1e-8]
        _, _, prefix, info = split_record(t, h, 1e-6)
        # The small A0-related trace consists solely of the early difference.
        self.assertLessEqual(info['delay_samples'], 2)
        self.assertEqual(len(prefix), info['delay_samples'])
        self.assertFalse(info['precursor_discarded'])

    def test_invalid_delay_and_changed_length_fail(self):
        t = np.arange(65)*.01
        h = np.zeros((65,2));h[2:] = .95**np.arange(63)[:,None]
        for d in (-1, 64, 1.5):
            with self.assertRaises(ValueError):
                split_record(t,h,delay_samples=d)
        model,_=fit_delayed(t,h,1,backend='numpy')
        with self.assertRaises(ValueError):
            delayed_impulse(model,1,64)


def physical_fixture():
    m = np.array([[2., .4, .1], [.4, 1.5, .2], [.1, .2, 1.]])
    k = np.array([[3100., -900., 300.], [-900., 2400., -350.], [300., -350., 1900.]])
    b = np.array([1., -.3, .2]); c = np.array([[.3, 1., -.2], [.3, 1.001, -.199]])
    return m,k,b,c


class MechanicalRemedyTests(unittest.TestCase):
    def test_actual_adapter_retains_consistent_body_mass_and_sparse_probes(self):
        m,k,b,c=physical_fixture()
        # One merged patch contains off-diagonal consistent mass. Replacement
        # of the diagonal contribution must occur once, not double-add it.
        storage=SimpleNamespace(cp=np,diagonal=np.diag(m),
            patches=[(np.arange(3),k,m,np.linalg.inv(m))],matvec=lambda a,x:a@x)
        def inverse(x):x[:]=np.linalg.solve(m,x)
        def launch(x,out,stiffness_scale,mass_scale):out[:]=stiffness_scale*k@x
        storage.inverse_mass_inplace=inverse;storage._launch=launch
        probes=SimpleNamespace(indices=[np.arange(3),np.arange(3)],weights=list(c))
        mass,inv,stiffness,observe=mechanical.coupled_actions(storage,probes)
        x=np.array([.2,-.7,.4])
        np.testing.assert_allclose(mass(x),m@x)
        np.testing.assert_allclose(inv(x.copy()),np.linalg.solve(m,x))
        np.testing.assert_allclose(stiffness(x),k@x)
        np.testing.assert_allclose(observe(x),c@x)

    def test_full_rank_matches_production_consistent_mass_verlet(self):
        m,k,b,c=physical_fixture(); n,dt=128,.001
        model,info=mechanical.project(3,b,lambda x:c@x,lambda x:m@x,
            lambda x:np.linalg.solve(m,x),lambda x:k@x,3)
        h,diagnostic=mechanical.impulse(model,3,n+1,dt)
        expected=simulate(m.tolist(),k.tolist(),b.tolist(),c.tolist(),n,dt,interior_impulse(dt))
        np.testing.assert_allclose(h,expected.displacement_m,rtol=1e-10,atol=1e-14)
        drive=np.random.default_rng(23).normal(size=n+1);drive[0]=0.
        direct=simulate(m.tolist(),k.tolist(),b.tolist(),c.tolist(),n,dt,lambda t:drive[round(t/dt)])
        np.testing.assert_allclose(reconstruct(h,drive),direct.displacement_m,rtol=1e-9,atol=1e-14)
        np.testing.assert_allclose(model['mass'],np.eye(3),atol=1e-12)
        self.assertTrue(diagnostic['marginal_oscillatory_stability_verified'])
        self.assertFalse(info['poles_clipped'])

    def test_every_nested_order_retains_positive_mass_and_stiffness(self):
        m,k,b,c=physical_fixture()
        model,_=mechanical.project(3,b,lambda x:c@x,lambda x:m@x,
            lambda x:np.linalg.solve(m,x),lambda x:k@x,3)
        for order in (1,2,3):
            _,info=mechanical.impulse(model,order,4097,.001)
            self.assertGreater(info['minimum_eigenvalue_s2'],0)
            self.assertGreater(info['verlet_stability_margin'],0)

    def test_invalid_physics_and_unstable_timestep_are_rejected(self):
        model=dict(mass=np.eye(2),stiffness=np.diag([2.,3.]),load=np.ones(2),observe=np.ones((2,2)))
        for key,value in [('mass',np.diag([-1.,1.])),('stiffness',np.diag([-1.,3.])),
                          ('stiffness',np.array([[2.,1.],[0.,3.]]))]:
            with self.assertRaises((ArithmeticError,np.linalg.LinAlgError)):
                mechanical.impulse(dict(model,**{key:value}),2,65,.1)
        with self.assertRaises(ArithmeticError):
            mechanical.impulse(model,2,65,2.)

    def test_krylov_closure_does_not_claim_requested_rank(self):
        model,info=mechanical.project(4,np.ones(4),lambda x:x[:2],lambda x:x.copy(),
                                     lambda x:x.copy(),lambda x:2*x,4)
        self.assertEqual(len(model['load']),1)
        self.assertTrue(info['numerical_krylov_closure'])

    def test_memory_guard_prevents_large_local_allocation(self):
        with patch.object(mechanical.psutil,'virtual_memory') as memory:
            memory.return_value.available=1
            with self.assertRaises(MemoryError):
                mechanical.project(3,np.ones(3),None,None,None,None,3)


class RemedySchedulerTests(unittest.TestCase):
    def test_tiny_complete_pipeline_and_tamper_detection(self):
        m,k,b,c=physical_fixture();n,dt=128,.001
        full=simulate(m.tolist(),k.tolist(),b.tolist(),c.tolist(),n,dt,interior_impulse(dt))
        t,h=full.time_s,full.displacement_m
        def fit(t,h,r,**kwargs):
            return fit_delayed(t,h,r,backend='numpy',prefix_relative_norm=kwargs['prefix_relative_norm'])
        def physical(*args,**kwargs):
            model,info=mechanical.project(3,b,lambda x:c@x,lambda x:m@x,
                lambda x:np.linalg.solve(m,x),lambda x:k@x,3,event=kwargs.get('event'))
            return model,dict(info,required_full_steps=n)
        source=lambda f:(lambda t:np.sin(f/1e4*t),{'tiny_fixture':True})
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(study,'DELAY_ORDERS',[6]),patch.object(study,'MECHANICAL_ORDERS',[3]),\
                 patch.object(study.baseline,'DT',dt),patch.object(study,'load_plan') as load,\
                 patch.object(study.baseline,'saved_kernel',return_value=(t,h)),\
                 patch.object(study.delay,'fit_delayed',side_effect=fit),\
                 patch.object(study.mechanical,'build_plate',side_effect=physical),\
                 patch.object(study.baseline,'target_source',side_effect=source),\
                 patch.object(study.baseline,'reference_trace',side_effect=lambda _,f:(t,reconstruct(h,source(f)[0](t)))):
                plan=dict(stages=study.stages(),configuration={})
                load.return_value=plan;write_json(root/'plan.json',plan)
                for stage in plan['stages']:
                    study.worker(root,stage)
                answer=study.assess(root)
                self.assertTrue(answer['completed'])
                self.assertEqual(len(answer['accepted_candidates']),2)
                self.assertTrue((root/'all-results.txt').exists())
                path=root/'mechanical-order-0003/50khz/waveforms.npz'
                path.write_bytes(b'corrupted')
                with self.assertRaises(ValueError):
                    study.assess(root)

    def test_missing_artifacts_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'mechanical-build';folder.mkdir()
            write_json(folder/'result.json',dict(completed=True,stage=folder.name,
                plan_sha256='x',artifacts_sha256={}))
            with self.assertRaises(ValueError):study.receipt(folder,'x')


if __name__=='__main__':unittest.main()
