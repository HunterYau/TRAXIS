"""Frozen wave settings and cooperative temperature control."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.embedded_wave import FULL_LEVELS, ThermalPause, case_config, checked_case, read_wave, screen
from examples.tensor_wave import sealed
from traxis.resources import ResourceLimitError
from traxis.storage import sha256_file, write_json


class EmbeddedWaveTests(unittest.TestCase):
    def test_declared_full_levels_fit_mesh_counts_and_preserve_physics(self):
        for defective in (False,True):
            reference=None
            for order,steps in FULL_LEVELS:
                case=case_config('full',order,steps,defective); checked_case(case)
                self.assertEqual(case['plate']['thickness_m'],.003175)
                self.assertEqual(case['duration_s'],.0005)
                self.assertLessEqual(screen(case)['vector_dofs'],12000000)
                fixed={k:v for k,v in case.items() if k not in ('order','steps')}
                if reference is not None: self.assertEqual(fixed,reference)
                reference=fixed

    def test_changed_geometry_pulse_limits_and_criteria_reject(self):
        case=case_config('full')
        for key,value in [('radius_m',.005),('energy_tolerance',1.),('cooling_upper_c',90.),
                           ('frequency_hz',150000.),('duration_s',.00025),('unexpected',1)]:
            changed=deepcopy(case); changed[key]=value
            with self.assertRaises(ValueError): checked_case(changed)
        changed=deepcopy(case); changed['plate']['thickness_m']=.003
        with self.assertRaises(ValueError): checked_case(changed)
        for args in [('full',4,65536,False),('full',True,None,False),('full',4,True,False),
                     ('smoke',3,None,False),('other',4,None,False),('smoke',4,None,1)]:
            with self.assertRaises(ValueError): case_config(*args)

    def test_smoke_time_levels_preserve_the_same_mechanical_case(self):
        a=case_config('smoke',steps=8064); b=case_config('smoke',steps=16128)
        self.assertEqual({k:v for k,v in a.items() if k!='steps'},{k:v for k,v in b.items() if k!='steps'})
        self.assertAlmostEqual(a['duration_s']/a['steps'],2*b['duration_s']/b['steps'])

    def test_cooling_uses_hysteresis_without_changing_physical_steps(self):
        class Guard:
            def __init__(self): self.last_gpus=[dict(temperature_c=76.)]; self.temperatures=iter([75.,73.,72.]); self.calls=0
            def check(self,force=False): self.calls+=1; self.last_gpus[0]['temperature_c']=next(self.temperatures)
        guard=Guard(); elapsed=[0.]
        def sleep(seconds): elapsed[0]+=seconds
        pacer=ThermalPause(guard,76.,72.,sleep=sleep,clock=lambda:elapsed[0])
        pacer.check(160)
        self.assertEqual(guard.calls,3); self.assertEqual(pacer.pause_seconds,.75)
        self.assertEqual(pacer.events[0]['steps_completed'],160)
        pacer.check(176); self.assertEqual(len(pacer.events),1)

    def test_cooling_keeps_resource_failure_and_missing_sensor_visible(self):
        class Guard:
            last_gpus=[dict(temperature_c=77.)]
            def check(self,force=False): raise ResourceLimitError('system RAM limit reached')
        pacer=ThermalPause(Guard(),76.,72.,sleep=lambda _:None)
        with self.assertRaises(ResourceLimitError): pacer.check(16)
        pacer.guard.last_gpus=[]
        with self.assertRaises(RuntimeError): pacer.check(32)


class EmbeddedRawReaderTests(unittest.TestCase):
    """Synthetic file fixtures exercise rejection, not mechanical accuracy."""
    def fixture(self, root, mutate=None):
        native_dir=root/'native'; native_dir.mkdir()
        case=case_config(); n=screen(case)['vector_dofs']; steps=case['steps']; dt=case['duration_s']/steps
        es=np.r_[np.arange(0,steps+1,case['energy_stride']),steps]
        arrays=dict(time_s=np.arange(steps+1)*dt,displacement_m=np.ones((steps+1,3)),energy_time_s=es*dt,
            physical_energy_j=np.ones(len(es)),modified_energy_j=np.ones(len(es)),
            final_displacement_m=np.zeros(n),final_velocity_m_s=np.zeros(n),mass_diagonal_kg=np.ones(n),final_gpu_force_n=np.zeros(n))
        patches={f'patch{i}_{key}':value for i in range(4) for key,value in
            [('indices',np.array([0],dtype=np.int64)),('weights',np.ones(1)),('direction',np.array([0.,0.,1.]))]}
        if mutate: mutate(arrays,patches)
        np.savez_compressed(native_dir/'mechanical-window.npz',**arrays)
        np.savez_compressed(native_dir/'mechanical-patches.npz',**patches)
        plan=sealed(dict(case=case)); write_json(root/'embedded-wave-plan.json',plan)
        native=sealed(dict(completed=True,passed=True,ready_for_training_data=False,training_updates=0,
            full_window_completed=True,plan_fingerprint=plan['fingerprint'],steps_completed=steps,planned_steps=steps,
            vector_dofs=n,dt_s=dt,patches=[dict(scalar_support=1) for _ in range(4)],energy_samples=len(es),
            propagation_statistics=dict(steps_completed=steps,requested_steps=steps,duration_s=case['duration_s']),
            mechanical_window_sha256=sha256_file(native_dir/'mechanical-window.npz'),
            mechanical_patches_sha256=sha256_file(native_dir/'mechanical-patches.npz')))
        write_json(native_dir/'wave-report.json',native)
        parent=sealed(dict(completed=True,passed=True,ready_for_training_data=False,training_updates=0,
            plan_fingerprint=plan['fingerprint'],worker_run=dict(exit_code=0),report_sha256=sha256_file(native_dir/'wave-report.json')))
        write_json(root/'monitor-result.json',parent)

    def test_bounded_raw_shapes_and_all_declared_samples_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.fixture(root); level=read_wave(root,states=True)
            self.assertEqual(level['window']['displacement_m'].shape,(8065,3))
            self.assertEqual(level['window']['time_s'][-1],.0000315)

    def test_changed_archive_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.fixture(root)
            with (root/'native/mechanical-window.npz').open('ab') as stream: stream.write(b'changed')
            with self.assertRaises(ValueError): read_wave(root)

    def test_rebound_bad_time_energy_trace_mass_and_patch_reject(self):
        cases=[lambda a,p:a['time_s'].__setitem__(1,1.),
               lambda a,p:a['modified_energy_j'].__setitem__(-1,2.),
               lambda a,p:a['displacement_m'].fill(0.),
               lambda a,p:a['mass_diagonal_kg'].__setitem__(0,-1.),
               lambda a,p:p['patch0_indices'].__setitem__(0,-1),
               lambda a,p:p['patch0_weights'].__setitem__(0,2.),
               lambda a,p:p.__setitem__('patch0_indices',np.array([0.],dtype=float)),
               lambda a,p:a.__setitem__('final_displacement_m',np.zeros(2))]
        for mutate in cases:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); self.fixture(root,mutate)
                with self.assertRaises(ValueError): read_wave(root,states=True)


if __name__=='__main__': unittest.main()
