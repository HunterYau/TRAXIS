"""Actual worker/queue publication pipeline with a synthetic integrator only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from examples import resolved_pzt_queue as queue
from traxis.storage import sha256_file


class RecordingPipelineTests(unittest.TestCase):
    def test_saved_waveforms_metrics_logs_summaries_and_serial_schedule(self):
        config={**queue.configuration(),'frequencies_hz':[50000,100000]}
        frozen={'configuration':config,'fingerprint':'synthetic full-publication check'}
        t=np.arange(1751)*2e-7
        channels=np.column_stack((np.exp(-((t-100e-6)/10e-6)**2)*np.cos(2*np.pi*50000*t),
                                 np.exp(-((t-170e-6)/10e-6)**2)*np.cos(2*np.pi*50000*t)))
        measured=np.column_stack((np.zeros((len(t),2)),channels))
        row={'nifgen_avg_waveform_ch1':np.array([0.,2.,0.]),'sampleRate':5000000,
             'niscope_avg_waveform':measured}
        system=SimpleNamespace(body_attachments=[],stable_dt_s=1e-8,
                               stiffness=np.zeros((1,1)),layout=SimpleNamespace(screen={'vector_dofs':1}))
        simulation=SimpleNamespace(time_s=t,displacement_m=channels,final_displacement_m=np.zeros(1),
            final_velocity_m_s=np.zeros(1),energy_time_s=np.linspace(0,350e-6,4),
            modified_energy_j=np.ones(4),statistics={'electrical_ports':[]})
        seen=[]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'prepared.json').write_text(json.dumps(frozen))
            def launch(command,project,env,guard,log,**kwargs):
                f=int(command[-2]);directory=Path(command[-1])
                if seen:
                    previous=root/f'{seen[-1]//1000}khz'
                    self.assertTrue((previous/'comparison.json').is_file())
                    self.assertIn('envelope Pearson r:',(root/'all-results.txt').read_text())
                seen.append(f)
                with log.open('w',encoding='utf-8') as stream,contextlib.redirect_stdout(stream):
                    queue.worker(f,directory)
                return {'exit_code':0}
            with patch.object(queue,'ROOT',root),patch.object(queue,'configuration',return_value=config), \
                    patch.object(queue,'preflight',return_value=frozen),patch.object(queue,'record',return_value=row), \
                    patch.object(queue,'build',return_value=(system,np.zeros(1),[],65536)), \
                    patch.object(queue,'gpu_explicit_simulate',return_value=simulation), \
                    patch.object(queue,'streamed_action',return_value=np.zeros(1)), \
                    patch.object(queue,'ThermalPause'), \
                    patch.object(queue,'PeakGuard',side_effect=lambda *a,**k:SimpleNamespace(peaks={})), \
                    patch.object(queue,'run_monitored',side_effect=launch),patch('sys.stdout',io.StringIO()):
                queue.run_queue()
                self.assertEqual(seen,[50000,100000])
                for f in seen:
                    directory=root/f'{f//1000}khz'
                    result=queue.validate_completed(f)
                    self.assertTrue(result['completed'])
                    self.assertAlmostEqual(result['comparison']['fast_envelope_pearson_r'],1.)
                    with np.load(directory/'waveforms.npz') as saved:
                        np.testing.assert_array_equal(saved['simulated_receiver_voltage_v'],channels)
                    self.assertIn('envelope Pearson r:',(root/f'summaries/{f//1000}khz.txt').read_text())
                    self.assertIn('trial_complete',(directory/'worker.log').read_text())
                    self.assertEqual(result['final_state_sha256'],sha256_file(directory/'final-state.npz'))
                self.assertEqual(len(json.loads((root/'all-results.json').read_text())),2)
                self.assertEqual(json.loads((root/'queue-state.json').read_text())['status'],'completed')


if __name__=='__main__':unittest.main()
