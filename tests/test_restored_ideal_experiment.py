"""Restored physical settings and worker wiring, with no GPU/full-plate run."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from examples import resolved_pzt_queue as queue


class RestoredIdealTests(unittest.TestCase):
    def test_physical_settings_match_the_saved_previous_ideal_experiment(self):
        config=queue.configuration()
        previous=json.loads((queue.PROJECT/config['restored_from']['prepared_receipt']).read_text())['configuration']
        for name in ('body','mesh','host_material','plate_m','material_profile','model_assumptions',
                     'actuator_center_xy_m','receiver_center_xy_m','disk_radius_m','disk_thickness_m'):
            self.assertEqual(config[name],previous[name],name)
        self.assertIsNone(queue.read_setup(config))
        self.assertEqual(queue.configured_damping(config,100000)['mass_damping_per_s'],0.)
        self.assertEqual(config['frequencies_hz'],[50000,100000,150000,200000,250000])
        self.assertFalse(config.get('ready_for_training_data'))

    def test_real_recorded_drive_uses_original_unit_peak_without_gain_or_delay(self):
        row=queue.record(100000)
        drive=np.asarray(row['nifgen_avg_waveform_ch1'])
        time=np.arange(len(drive))/row['sampleRate']
        actual=queue.actuator_signal(None,time,drive)
        np.testing.assert_array_equal(actual(time),drive/np.max(abs(drive)))
        self.assertEqual(actual(-1.),0.)
        self.assertEqual(actual(1.),0.)
        self.assertEqual(queue.receiver_ports(object(),None),[])

    def test_perfect_body_build_adds_no_adhesive_or_instrument_capacitance(self):
        config=queue.configuration();bodies=[]
        def attach(*args,**kwargs):
            index=kwargs['first_dof']
            body=SimpleNamespace(next_dof=index+1,indices=np.array([index]),coupling=np.ones(1),
                                 total_capacitance_f=1.,report={})
            bodies.append(body)
            return body
        layout=SimpleNamespace(screen={'vector_dofs':2})
        system=SimpleNamespace(stable_dt_s=1e-8,screen={'device_storage_screen_mib':0})
        with patch.object(queue,'make_layout',return_value=layout), \
                patch.object(queue,'build_gpu_tensor'), \
                patch.object(queue,'attach_disk',side_effect=attach) as attachment, \
                patch.object(queue,'couple_gpu_bodies',return_value=system), \
                patch.object(queue,'sparse_probe_set'):
            queue.build(config,None,lambda *a,**k:None)
        self.assertEqual(attachment.call_count,3)
        for call in attachment.call_args_list:
            self.assertEqual(call.kwargs['bond'],'perfect')
            self.assertIsNone(call.kwargs['adhesive'])
            self.assertEqual(call.kwargs['input_capacitance_f'],0.)

    def test_worker_passes_ideal_settings_to_integrator(self):
        config=queue.configuration()
        frozen={'configuration':config,'fingerprint':'synthetic wiring check'}
        system=SimpleNamespace(body_attachments=[],stable_dt_s=1e-8)
        row={'nifgen_avg_waveform_ch1':np.array([0.,2.,0.]),'sampleRate':5000000}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'prepared.json').write_text(json.dumps(frozen))
            with patch.object(queue,'ROOT',root), \
                    patch.object(queue,'preflight',return_value=frozen), \
                    patch.object(queue,'build',return_value=(system,np.zeros(1),[],65536)), \
                    patch.object(queue,'record',return_value=row), \
                    patch.object(queue,'PeakGuard',return_value=SimpleNamespace(peaks={})), \
                    patch.object(queue,'gpu_explicit_simulate',return_value=SimpleNamespace(statistics={})) as solver, \
                    patch('sys.stdout',io.StringIO()):
                queue.worker(50000,root/'50khz',smoke_steps=128)
            self.assertEqual(solver.call_args.kwargs['mass_damping_per_s'],0.)
            self.assertEqual(solver.call_args.kwargs['electrical_ports'],[])
            self.assertEqual(solver.call_args.kwargs['force'](.2e-6),1.)
            saved=json.loads((root/'50khz/result.json').read_text())
            self.assertIsNone(saved['assumed_setup'])
            self.assertTrue(saved['smoke_test'])
            self.assertFalse((root/'50khz/waveforms.npz').exists())
            self.assertTrue((root/'50khz/summary.txt').is_file())
            self.assertFalse((root/'summaries/50khz.txt').exists())


if __name__=='__main__':
    unittest.main()
