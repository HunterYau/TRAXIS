"""Small real-GPU parity checks; no full plate or experimental simulations."""
import unittest
from unittest.mock import patch
import numpy as np
from traxis.config import Plate,Material
from traxis.gpu_tensor import make_layout,build_gpu_tensor
from traxis.gpu_local_time import gpu_lts_simulate as original,LocalActions
from traxis.gpu_lts_optimized import gpu_lts_simulate as optimized,ReusableActions
from traxis.gpu_enriched_verlet import sparse_probe_set
from traxis.local_time import fine_partition


class OptimizedLTSTests(unittest.TestCase):
    measurements=[]
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount()<1:raise RuntimeError('No GPU')
        except Exception as exc:raise unittest.SkipTest(str(exc))
        cls.cp=cp
        cls.layout=make_layout(Plate(.04,.04,.001),(3,3,1),2)
        cls.system=build_gpu_tensor(cls.layout,Material())

    def compare(self,system,load,probes,*,sampling='coarse',steps=67,graph_steps=7,unchecked=False):
        fine=fine_partition(system,load)
        H=5.340576171875e-9 if unchecked else .3*system.stable_dt_s
        # Two separated pulses ensure the graph reads changing source/step data.
        source=lambda t:float(abs(t/H-1)<.3)+.37*float(abs(t/H-4.5)<.3)
        kwargs=dict(fine=fine,force=source,source_sampling=sampling,energy_stride=11)
        reference=original
        if unchecked:
            from traxis.gpu_local_time_unchecked import gpu_lts_simulate as reference
        ta,ya,sa=reference(system,steps,H,probes,load,**kwargs)
        for graphs in (False,True):
            events=[]
            tb,yb,sb=optimized(system,steps,H,probes,load,**kwargs,
                use_graphs=graphs,graph_steps=graph_steps,progress=events.append,screen_stability=not unchecked)
            np.testing.assert_array_equal(ta,tb)
            relative=np.linalg.norm(yb-ya)/np.linalg.norm(ya)
            self.assertLess(relative,1e-9)
            from scipy.signal import hilbert
            derived=lambda y: np.column_stack((y,y[:,0]+y[:,-1],y[:,0]-y[:,-1]))
            errors=[];envelope_errors=[]
            for x,z in zip(derived(ya).T,derived(yb).T):
                if np.linalg.norm(x)>1e-20:
                    errors.append(float(np.linalg.norm(z-x)/np.linalg.norm(x)))
                    envelope_errors.append(float(np.linalg.norm(abs(hilbert(z))-abs(hilbert(x)))/np.linalg.norm(x)))
            if errors:self.assertLess(max(errors),1e-8)
            if envelope_errors:self.assertLess(max(envelope_errors),1e-8)
            ea=np.array(sa['modified_energy_j']);eb=np.array(sb['modified_energy_j'])
            self.assertLess(np.linalg.norm(eb-ea)/np.linalg.norm(ea),1e-9)
            self.assertEqual(sa['energy_time_s'],sb['energy_time_s'])
            self.assertEqual([x['steps_completed'] for x in events],list(range(16,steps,16))+[steps])
            self.assertEqual(sb['cuda_graphs'],graphs)
            self.assertEqual(sb['stability_screen'],sa['stability_screen'])
            if graphs:self.assertGreater(sb['graph_launches'],1)
            self.measurements.append(dict(dofs=len(load),source_sampling=sampling,steps=steps,graphs=graphs,
                coarse_dt_s=H,fine_dt_s=H/2,stability_screened=not unchecked,
                voltage_or_displacement_relative_l2=float(relative),
                largest_channel_relative_l2=max(errors,default=0),
                largest_envelope_relative_l2=max(envelope_errors,default=0),
                energy_relative_l2=float(np.linalg.norm(eb-ea)/np.linalg.norm(ea)),
                graph_variants=sb['graph_variants'],graph_launches=sb['graph_launches']))

    def test_plate_coarse_and_fine_source_graph_replay_and_remainders(self):
        n=len(self.system.mass_diagonal);load=np.zeros(n);load[0]=1
        probes=sparse_probe_set([(np.array([0,13]),np.array([1.,-.2])),
                                 (np.array([n//2,n-1]),np.array([1.,.7]))],n)
        for sampling in ('coarse','fine'):
            self.compare(self.system,load,probes,sampling=sampling)
        self.compare(self.system,load,probes,sampling='coarse',steps=513,graph_steps=16)

    def test_body_mass_coupling_actions_and_voltage_parity(self):
        from tests.test_pzt_receiver import material
        from traxis.pzt_body import attach_disk
        from traxis.pzt_receiver import DiskGeometry
        from traxis.gpu_pzt_body import couple_gpu_bodies
        layout=make_layout(Plate(.04,.04,.001),(2,2,1),2)
        base=build_gpu_tensor(layout,Material());bodies=[];next_dof=len(base.mass_diagonal)
        for center,surface,electrical in (((.01,.02),'top','voltage'),((.03,.02),'top','open'),((.03,.02),'bottom','open')):
            b=attach_disk(layout,material(),DiskGeometry(center,.004,.0005,surface,1),
                sectors=8,rings=1,layers=1,first_dof=next_dof,electrical=electrical)
            bodies.append(b);next_dof=b.next_dof
        system=couple_gpu_bodies(base,bodies);load=np.zeros(next_dof)
        load[bodies[0].indices]=-bodies[0].coupling
        probes=sparse_probe_set([(b.indices,b.coupling/b.total_capacitance_f) for b in bodies[1:]],next_dof)
        fine=fine_partition(system,load);old=LocalActions(system,fine);new=ReusableActions(system,fine)
        x=self.cp.asarray(np.random.default_rng(123).normal(size=next_dof));out=self.cp.empty_like(x)
        new.full_into(x,out)
        np.testing.assert_allclose(self.cp.asnumpy(out),self.cp.asnumpy(old.full(x)),rtol=2e-11,atol=.5)
        new.local_into(self.cp.where(new.fine,x,0.),out)
        np.testing.assert_allclose(self.cp.asnumpy(out),self.cp.asnumpy(old.local(x)),rtol=2e-11,atol=.5)
        self.compare(system,load,probes,sampling='coarse',steps=65,graph_steps=16)
        self.compare(system,load,probes,sampling='fine',steps=33,graph_steps=3)
        self.compare(system,load,probes,sampling='coarse',steps=257,graph_steps=16)
        with patch('traxis.gpu_lts_optimized.lts_stability_screen',side_effect=AssertionError('Unexpected screen')):
            self.compare(system,load,probes,sampling='coarse',steps=257,graph_steps=16,unchecked=True)

    def test_single_step_and_original_ratio_fallback(self):
        n=len(self.system.mass_diagonal);load=np.zeros(n);load[0]=1
        probes=sparse_probe_set([(np.array([0]),np.ones(1))],n)
        fine=fine_partition(self.system,load);H=.1*self.system.stable_dt_s
        for p in (1,2):
            kw=dict(fine=fine,force=lambda t:float(t>0),substeps=p)
            t,a,_=original(self.system,1,H,probes,load,**kw)
            t,b,_=optimized(self.system,1,H,probes,load,**kw)
            np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-25)
        with self.assertRaises(ValueError):optimized(self.system,1,H,probes,load,fine=fine,graph_steps=0)
        with self.assertRaises(ValueError):optimized(self.system,1,H,probes,load,fine=fine,mass_damping_per_s=1.)


if __name__=='__main__':unittest.main()
