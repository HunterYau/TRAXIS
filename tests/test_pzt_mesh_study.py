"""Lightweight protocol/geometry/report/recovery checks. No full simulations."""
import contextlib
import csv
import importlib.util
import io
import json
import shutil
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from examples import pzt_mesh_protocol as protocol,pzt_mesh_metrics as metrics,pzt_mesh_study as study
from traxis.config import Plate
from traxis.gpu_tensor import make_layout
from traxis.pzt_body import disk_mesh,attach_disk
from traxis.pzt_receiver import DiskGeometry
from tests.test_pzt_receiver import material
from tools.runpod import pzt_mesh_transfer as transfer


class StudyTests(unittest.TestCase):
    def test_optimized_primary_worker_preserves_steps_check_policy_and_outputs(self):
        row=dict(protocol.cases()[0],estimate=dict(total_mechanical_dofs=24))
        H=protocol.REQUESTED_COARSE_DT_S
        frequencies=[50000,100000,150000,200000,250000,300000]
        plan=dict(cases=[row],configuration=dict(resource_limits={}),edge_width_m=0.,boxes=[],duration_s=33*H,
            execution=dict(primary_only=True,skip_stability_checks=True,fixed_coarse_dt_s=H,backend='optimized'),
            protocol=dict(device_budget_mib=1024,frequencies_hz=frequencies))
        system=SimpleNamespace(mass_diagonal=np.ones(24),screen={},body_attachments=[],stable_dt_s=1e-12)
        guard=SimpleNamespace(check=lambda **kw:None,peaks={},telemetry_error=None)
        def solver(system,steps,h,*args,**kw):
            self.assertEqual(steps,33);self.assertEqual(h,H)
            self.assertEqual(kw['substeps'],2);self.assertEqual(kw['source_sampling'],'coarse')
            self.assertTrue(kw['use_graphs']);self.assertFalse(kw['screen_stability'])
            y=np.column_stack((np.arange(steps+1),-np.arange(steps+1))).astype(float);y[:2]=0
            return np.arange(steps+1)*h,y,dict(energy_time_s=[10*H,20*H],modified_energy_j=[1.,1.])
        def analyze(folder,*args):
            study.save(folder/'comparisons.json',dict(temporal_qualified=False))
            for f in frequencies:study.save(folder/'plots'/f'{f//1000}khz'/'vs-baseline/plots.json',dict(files={}))
            return dict(temporal_qualified=False)
        with tempfile.TemporaryDirectory() as tmp,patch.object(protocol,'validate',return_value=plan),\
             patch.object(study,'build',return_value=(system,np.ones(24),None)),\
             patch.object(study,'fine_partition',return_value=np.ones(24,dtype=bool)),\
             patch('traxis.gpu_lts_optimized.gpu_lts_simulate',side_effect=solver) as simulate,\
             patch.object(study,'optimized_stability_screen',side_effect=AssertionError('Unexpected screen')),\
             patch.object(study,'analysis',side_effect=analyze),patch.object(study,'PeakGuard',return_value=guard),\
             contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);study.worker(root,row['id']);folder=root/'cases'/row['id']
            self.assertTrue(study.completed(folder));self.assertFalse(study.read(folder/'result.json')['stability_checks_performed'])
            self.assertEqual(len(list((folder/'primary').glob('*khz/waveforms.npz'))),6)
            study.worker(root,row['id']);self.assertEqual(simulate.call_count,1)

    def test_unchecked_fixed_steps_never_screen_or_reduce_step(self):
        H=.00035/65536
        plan=dict(execution=dict(skip_stability_checks=True,fixed_coarse_dt_s=H),duration_s=.00035)
        # A bound below the requested step must not trigger a guard or step reduction.
        with patch.object(study,'optimized_stability_screen',side_effect=AssertionError('Screen called')):
            result=study.select_time(plan,SimpleNamespace(stable_dt_s=1e-12),None,lambda *a,**k:None,'auto',None)
        self.assertEqual(result['coarse_dt_s'],H)
        self.assertEqual(result['fine_dt_s'],H/2)
        self.assertEqual(result['primary_steps'],65536)
        self.assertEqual(result['half_step_steps'],0)
        self.assertIsNone(result['stability']['accepted'])
        self.assertFalse(result['stability']['performed'])

    def test_scheduler_stops_before_next_configuration_after_failure(self):
        rows=protocol.cases();plan=dict(cases=rows,execution={})
        with tempfile.TemporaryDirectory() as tmp,patch.object(protocol,'validate',return_value=plan),patch.object(study,'completed',return_value=False):
            root=Path(tmp)
            study.save(root/'cases'/rows[0]['id']/'result.json',dict(completed=False,status='failed',configuration=rows[0],error='failed screen'))
            report=dict(configurations=[dict(configuration=r,status='failed' if r['level']==1 else 'pending',completed=False) for r in rows])
            with patch.object(study,'assess',return_value=report),patch.object(study.subprocess,'Popen') as launch:
                study.run(root)
                launch.assert_not_called()
            status=study.read(root/'status.json')
            self.assertEqual(status['status'],'stopped_after_failure')
            self.assertEqual(status['pending'],[r['id'] for r in rows[1:]])
            self.assertEqual(status['failed'],[rows[0]['id']])

    def test_exact_2_combined_configurations_no_baseline_or_individual_jobs(self):
        rows=protocol.cases();self.assertEqual(len(rows),2)
        for r in rows:
            self.assertIn(r['level'],(1,2))
            self.assertEqual(r['family'],'combined')
            wanted=protocol.LEVELS[r['level']-1]
            self.assertEqual({k:r[k] for k in protocol.BASE},wanted)
            self.assertEqual(json.loads((protocol.CONFIG/(r['id']+'.json')).read_text()),r)
        self.assertFalse(any(r['id']=='baseline' for r in rows))

    def test_requested_meshes_valid_without_dense_assembly(self):
        geometry=DiskGeometry((.5,.5),.005,.0005,'top',1)
        for r in protocol.cases():
            mesh,face=disk_mesh(geometry,sectors=r['angular'],rings=r['radial'],layers=r['thickness'],max_nodes=8192,max_elements=65536)
            self.assertEqual(len(mesh.points_m),(1+r['angular']*r['radial'])*(r['thickness']+1))
            self.assertEqual(len(mesh.tetrahedra),3*r['thickness']*r['angular']*(2*r['radial']-1))
            t=mesh.points_m[mesh.tetrahedra]
            volume=np.abs(np.linalg.det((t[:,1:]-t[:,0,None]).transpose(0,2,1))).sum()/6
            expected=r['angular']*.5*.005**2*np.sin(2*np.pi/r['angular'])*.0005
            self.assertAlmostEqual(volume/expected,1.,places=12)
        with self.assertRaises(ValueError):disk_mesh(geometry,sectors=32,rings=8,layers=3)

    def test_baseline_assembly_exactly_matches_preserved_original(self):
        paths=[protocol.PROJECT/'outputs/simulation-runs/pzt-mesh-convergence-002/baseline-assembly-sources',
               protocol.PROJECT/'outputs/pzt-mesh-preparation-audit/baseline']
        old=next(p for p in paths if (p/'pzt_body.py').exists())
        name='pzt_mesh_legacy_body';spec=importlib.util.spec_from_file_location(name,old/'pzt_body.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        layout=make_layout(Plate(.04,.04,.001),(2,2,1),2)
        args=(layout,material(),DiskGeometry((.02,.02),.005,.0005,'top',1))
        a=module.attach_disk(*args,sectors=8,rings=1,layers=1,max_workspace_mib=64)
        b=attach_disk(*args,sectors=8,rings=1,layers=1,max_workspace_mib=64,max_component_nodes=8192,max_component_elements=65536)
        for name in ('stiffness','mass','coupling','indices','points_m'):np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
        for name in ('pzt_body.py','piezoelectric.py'):
            self.assertEqual(protocol.mathematical_ast((old/name).read_text()),
                             protocol.mathematical_ast((protocol.PROJECT/'traxis'/name).read_text()))

    def waves(self):
        t=np.arange(2049)*(.00035/2048)
        x=np.sin(2*np.pi*50000*t)*np.sin(np.pi*t/t[-1])**2
        return t,np.column_stack((x,.2*x))

    def test_raw_amplitude_wiggle_and_envelope_are_distinct(self):
        t,y=self.waves()
        config=dict(actuator_center_xy_m=[.5,.5],receiver_center_xy_m=[.75,.5],plate_m=[1,1,.001],
                    host_material=dict(young_pa=71e9,poisson=.33,density_kg_m3=2700))
        result=metrics.compare(t,y,t,2*y,50000,config,arrivals=False)
        for row in result['channels'].values():
            self.assertAlmostEqual(row['raw_l2_percent'],100.,places=10)
            self.assertAlmostEqual(row['envelope_l2_percent'],100.,places=10)
            self.assertAlmostEqual(row['wiggle_l2_percent'],0.,places=10)
            self.assertAlmostEqual(row['raw_pearson'],1.,places=12)

    def test_nested_nodes_and_undefined_signals_cannot_pass(self):
        t,y=self.waves();shared,a,b=metrics.shared_grid(t[::2],y[::2],t,y)
        np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):metrics.shared_grid(t,y,t+1e-8,y)
        self.assertFalse(metrics.temporal_pass([]))
        undefined=[dict(channels={'top':dict(raw_l2_percent=None,envelope_l2_percent=0.,wiggle_l2_percent=0.)})]
        self.assertFalse(metrics.temporal_pass(undefined))

    def test_existing_headless_plotting_exports_png_and_svg(self):
        t,y=self.waves()
        with tempfile.TemporaryDirectory() as tmp:
            paths=metrics.plots(Path(tmp),t,y,t,1.001*y,'Lightweight test only')
            self.assertEqual(set(paths),{'waveforms.png','waveforms.svg'})
            for p in paths:self.assertGreater((Path(tmp)/p).stat().st_size,1000)

    def test_step_selection_halves_only_when_needed_and_respects_solver_limit(self):
        system=SimpleNamespace(stable_dt_s=1.)
        plan=dict(baseline_coarse_dt_s=1.,duration_s=32768.)
        with patch.object(study,'optimized_stability_screen',side_effect=[ArithmeticError('LTS coarse step fails effective-operator stability screen'),{'accepted':True}]):
            p=study.select_time(plan,system,np.ones(1,dtype=bool),lambda *a,**k:None,'auto',None)
        self.assertEqual(p['coarse_dt_s'],.5);self.assertEqual(p['half_step_steps'],131072)
        system.stable_dt_s=.01
        with self.assertRaises(ValueError):study.select_time(plan,system,np.ones(1,dtype=bool),lambda *a,**k:None,'auto',None)

    def test_interrupted_half_step_reuses_completed_primary(self):
        row=dict(protocol.cases()[0],estimate=dict(total_mechanical_dofs=24))
        limits={name:None for name in ('max_seconds','max_rss_mib','max_system_memory_percent','min_available_mib',
            'max_gpu_memory_percent','max_gpu_temperature_c','max_cpu_temperature_c')}
        plan=dict(cases=[row],configuration=dict(resource_limits=limits),edge_width_m=.004,boxes=[],
            protocol=dict(device_budget_mib=1024,frequencies_hz=[50000,100000,150000,200000,250000,300000]))
        system=SimpleNamespace(mass_diagonal=np.ones(24),screen={},body_attachments=[],stable_dt_s=1.)
        settings=dict(coarse_dt_s=.00035/64,primary_steps=64)
        calls=[];fail=True
        def fake_solver(system,steps,H,*a,**kw):
            nonlocal fail
            calls.append(steps)
            if steps==128 and fail:fail=False;raise RuntimeError('Simulated half-step interruption')
            t=np.arange(steps+1)*H;y=np.column_stack((np.sin(t*1e5),np.cos(t*1e5)));y[:2]=0
            return t,y,dict(energy_time_s=[10*H,20*H],modified_energy_j=[1.,1.])
        def fake_analysis(folder,p,r,root):
            study.save(folder/'comparisons.json',dict(temporal_qualified=True))
            for f in p['protocol']['frequencies_hz']:
                for relation in ('vs-baseline','temporal'):
                    study.save(folder/'plots'/f'{f//1000}khz'/relation/'plots.json',dict(files={}))
            return dict(temporal_qualified=True)
        guard=SimpleNamespace(check=lambda **kw:None,peaks={},telemetry_error=None)
        with tempfile.TemporaryDirectory() as tmp,patch.object(study.protocol,'validate',return_value=plan),\
             patch.object(study,'build',return_value=(system,np.ones(24),None)),\
             patch.object(study,'select_time',return_value=settings),\
             patch.object(study,'fine_partition',return_value=np.ones(24,dtype=bool)),\
             patch.object(study,'optimized_stability_screen',return_value={'model_fingerprint':'test-only'}),\
             patch.object(study,'gpu_lts_simulate',side_effect=fake_solver),\
             patch.object(study,'analysis',side_effect=fake_analysis),patch.object(study,'PeakGuard',return_value=guard),\
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):study.worker(Path(tmp),row['id'])
            self.assertTrue(study.read(Path(tmp)/'cases'/row['id']/'primary/result.json')['completed'])
            study.worker(Path(tmp),row['id'])
            self.assertTrue(study.completed(Path(tmp)/'cases'/row['id']))
            with (Path(tmp)/'cases'/row['id']/'primary/trace.npz').open('ab') as f:f.write(b'corrupt')
            with self.assertRaises(ValueError):study.completed(Path(tmp)/'cases'/row['id'])
        self.assertEqual(calls,[64,128,128])

    def test_failed_case_reports_all_frequencies_and_channels_without_fake_scores(self):
        plan=dict(cases=protocol.cases(),protocol=dict(frequencies_hz=[50000,100000,150000,200000,250000,300000]))
        with tempfile.TemporaryDirectory() as tmp,patch.object(study.protocol,'validate',return_value=plan):
            root=Path(tmp);folder=root/'cases'/plan['cases'][0]['id']
            study.save(folder/'result.json',dict(completed=False,status='failed',configuration=plan['cases'][0],error='test failure'))
            report=study.assess(root)
            rows=list(csv.DictReader((root/'metrics.csv').open()))
            self.assertEqual(len(report['configurations']),2);self.assertEqual(len(rows),24)
            self.assertEqual({r['channel'] for r in rows},set(metrics.CHANNELS))
            self.assertTrue(all(r['status']=='failed' and r['raw_l2_percent']=='' for r in rows))

    def test_analysis_records_all_channels_and_creates_plot_receipts(self):
        row=protocol.cases()[0];t,y=self.waves()
        config=dict(actuator_center_xy_m=[.5,.5],receiver_center_xy_m=[.75,.5],plate_m=[1,1,.001],
                    host_material=dict(young_pa=71e9,poisson=.33,density_kg_m3=2700))
        frequencies=[50000,100000,150000,200000,250000,300000]
        plan=dict(cases=protocol.cases(),configuration=config,
                  protocol=dict(frequencies_hz=frequencies,temporal_target_percent=.01))
        def plot(folder,*a):
            folder.mkdir(parents=True);(folder/'test-plot.svg').write_text('<svg/>')
            return ['test-plot.svg']
        with tempfile.TemporaryDirectory() as tmp,patch.object(metrics,'plots',side_effect=plot):
            root=Path(tmp);folder=root/'cases'/row['id']
            for f in frequencies:
                for parent in (root/'baseline/lts-broadband',folder/'primary',folder/'half-step'):
                    dest=parent/f'{f//1000}khz';dest.mkdir(parents=True)
                    np.savez(dest/'waveforms.npz',time_s=t,voltage_v=y)
            result=study.analysis(folder,plan,row,root)
            self.assertTrue(result['temporal_qualified'])
            self.assertEqual(len(result['vs_baseline']),6)
            self.assertEqual(set(result['vs_baseline'][0]['channels']),set(metrics.CHANNELS))
            self.assertEqual(len(list((folder/'plots').rglob('plots.json'))),12)
            study.analysis(folder,plan,row,root) # Existing verified plots are reused.

    def test_download_watcher_verifies_export_and_leaves_pod_untouched(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            project=Path(tmp)/'TRAXIS';root=project/'outputs/simulation-runs/test';root.mkdir(parents=True)
            (root/'status.json').write_text(json.dumps(dict(status='completed_with_failures')))
            (root/'all-results.txt').write_text('Retain unsuccessful configurations too.\n')
            captured=io.StringIO()
            with patch.object(transfer,'PROJECT',project),contextlib.redirect_stdout(captured):transfer.export(root)
            info=json.loads(captured.getvalue());archive=Path(info['archive'])
            remote='/workspace/test/TRAXIS';remote_info=dict(info,archive=remote+'/outputs/pzt-mesh-exports/'+archive.name)
            key=Path(tmp)/'private';known=Path(tmp)/'known';key.write_text('test');known.write_text('test')
            args=SimpleNamespace(host='test.example',port=22,remote_project=remote,
                remote_root='outputs/simulation-runs/test',output=Path(tmp)/'download',detach=False,key=key,known_hosts=known)
            commands=[]
            def ssh(command,**kw):
                commands.append(command[-1])
                return json.dumps(dict(status=dict(status='completed_with_failures'),alive=False) if len(commands)==1 else remote_info)
            def scp(command,**kw):shutil.copy2(archive,command[-1])
            with patch.object(transfer.subprocess,'check_output',side_effect=ssh),patch.object(transfer.subprocess,'run',side_effect=scp):
                transfer.watch(args)
            receipt=json.loads((args.output/'download-verified.json').read_text())
            self.assertTrue(receipt['pod_left_running']);self.assertEqual(len(commands),2)
            self.assertTrue(commands[0].endswith('status --root outputs/simulation-runs/test'))
            self.assertTrue(commands[1].endswith('export --root outputs/simulation-runs/test'))
            self.assertEqual((Path(receipt['destination'])/'study/all-results.txt').read_text(),
                             (root/'all-results.txt').read_text())


if __name__=='__main__':unittest.main()
