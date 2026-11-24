"""Tiny actual CUDA comparisons; no experimental/full-plate runs."""
import unittest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from types import SimpleNamespace
import io,json
from traxis.config import Plate,Material
from traxis.gpu_tensor import make_layout,build_gpu_tensor,streamed_action
from traxis.gpu_local_time import LocalActions,gpu_lts_simulate
from traxis.local_time import fine_partition
from traxis.gpu_enriched_verlet import sparse_probe_set,gpu_explicit_simulate


class GPULocalTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount()<1: raise RuntimeError('no CUDA device')
        except Exception as exc: raise unittest.SkipTest(str(exc))
        cls.cp=cp
        cls.layout=make_layout(Plate(.04,.03,.003),(3,2,1),2)
        cls.system=build_gpu_tensor(cls.layout,Material())

    def test_selected_cuda_interface_rows_match_independent_cpu_action(self):
        n=len(self.system.mass_diagonal);fine=np.zeros(n,dtype=bool);fine[0]=True
        x=np.random.default_rng(12).normal(size=n)
        actions=LocalActions(self.system,fine)
        expected=streamed_action(self.layout,Material(),x*fine)/self.system.mass_diagonal
        actual=self.cp.asnumpy(actions.local(self.cp.asarray(x)))
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-3)
        self.assertLess(len(actions.ids),self.layout.screen['elements'])
        self.assertTrue(np.any(actual[~fine]!=0))

    def test_p1_gpu_matches_existing_verlet(self):
        n=len(self.system.mass_diagonal);load=np.zeros(n);load[0]=1.
        probes=sparse_probe_set([(np.array([0,1]),np.array([1.,-.3]))],n)
        fine=fine_partition(self.system,load);H=.1*self.system.stable_dt_s
        source=lambda t:np.sin(t/H*.1)
        t,y,_=gpu_lts_simulate(self.system,64,H,probes,load,fine=fine,substeps=1,force=source)
        old=gpu_explicit_simulate(self.system,64,H,probes,load,force=source,energy_stride=16)
        np.testing.assert_allclose(y,old.displacement_m,rtol=1e-10,atol=1e-20)

    def test_reject_damping_and_unstable_coarse_step(self):
        n=len(self.system.mass_diagonal);load=np.zeros(n);load[0]=1.
        fine=fine_partition(self.system,load);probes=sparse_probe_set([(np.array([0]),np.ones(1))],n)
        with self.assertRaises(ValueError):gpu_lts_simulate(self.system,1,self.system.stable_dt_s,probes,load,fine=fine)
        with self.assertRaises(ValueError):gpu_lts_simulate(self.system,1,.1*self.system.stable_dt_s,probes,load,fine=fine,mass_damping_per_s=1.)

    def test_resolved_body_local_action_preserves_consistent_mass(self):
        from tests.test_pzt_receiver import material
        from traxis.pzt_body import attach_disk
        from traxis.pzt_receiver import DiskGeometry
        from traxis.gpu_pzt_body import couple_gpu_bodies
        layout=make_layout(Plate(.04,.04,.001),(2,2,1),2)
        base=build_gpu_tensor(layout,Material())
        body=attach_disk(layout,material(),DiskGeometry((.02,.02),.005,.0005,'top',1),
                         sectors=8,rings=1,layers=1,electrical='open')
        system=couple_gpu_bodies(base,[body])
        n=len(system.mass_diagonal);load=np.zeros(n);load[-1]=1
        fine=fine_partition(system,load);actions=LocalActions(system,fine)
        x=np.random.default_rng(13).normal(size=n)*fine
        force=np.zeros(n);host=layout.screen['vector_dofs']
        force[:host]=streamed_action(layout,Material(),x[:host])
        for patch in system.body_patches:force[patch.indices]+=patch.stiffness@x[patch.indices]
        expected=force/system.mass_diagonal
        for patch in system.body_patches:expected[patch.indices]=np.linalg.solve(patch.mass,force[patch.indices])
        actual=self.cp.asnumpy(actions.local(self.cp.asarray(x)))
        np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=.5)
        bad=fine.copy();bad[system.body_patches[0].indices[0]]=False
        probes=sparse_probe_set([(np.array([0]),np.ones(1))],n)
        with self.assertRaises(ValueError):gpu_lts_simulate(system,1,.1*system.stable_dt_s,probes,load,fine=bad)

    def test_serial_schedule_and_resume_with_real_tiny_gpu_solver(self):
        from examples import pzt_local_time_study as study
        from traxis.storage import write_json
        n=len(self.system.mass_diagonal);load=np.zeros(n);load[0]=1
        probes=sparse_probe_set([(np.array([0]),np.ones(1)),(np.array([1]),np.ones(1))],n)
        H=.1*self.system.stable_dt_s
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan=dict(identity={},configuration=dict(mesh=dict(order=2),body={},resource_limits={}),
                case=dict(cells=3,body=[8,1,1],steps=64),substeps=2,coarse_dt_s=H,
                boxes=[],edge_width_m=0,frequencies_hz=[100000],
                tasks=['kernel-1','kernel-2','uniform-100000','lts-100000'])
            write_json(root/'plan.json',plan)
            with patch.object(study,'identity',return_value={}),patch.object(study.queue,'build',return_value=(self.system,load,probes,64)),patch.object(study,'PeakGuard',return_value=SimpleNamespace(check=lambda **kw:None)),patch.object(study.baseline,'source',return_value=(lambda t:np.sin(t/H*.1),{})),redirect_stdout(io.StringIO()):
                study.run(root)
                before=[(root/t/'trace.npz').stat().st_mtime_ns for t in plan['tasks']]
                study.run(root)
                after=[(root/t/'trace.npz').stat().st_mtime_ns for t in plan['tasks']]
            self.assertEqual(before,after)
            self.assertEqual(json.loads((root/'status.json').read_text())['status'],'completed')
            rows=json.loads((root/'comparisons.json').read_text())
            self.assertEqual(len(rows),1)


if __name__=='__main__':unittest.main()
