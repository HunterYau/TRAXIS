"""Long-run gate and automatic measured-comparison reporting."""
import unittest
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from examples.resolved_pzt_queue import compare,run_queue,preflight,configuration,PROJECT
from examples.pzt_terminal import watch
from traxis.storage import sha256_file


class QueueTests(unittest.TestCase):
    def test_held_queue_cannot_spawn_or_claim_ownership(self):
        with patch('examples.resolved_pzt_queue.configuration',return_value={'launch_ready':False}), \
                patch('examples.resolved_pzt_queue.write_json') as write, \
                patch('examples.resolved_pzt_queue.run_monitored') as run:
            with self.assertRaisesRegex(RuntimeError,'Long queue held'):run_queue()
        write.assert_not_called();run.assert_not_called()

    def test_readiness_cannot_be_enabled_with_altered_proof(self):
        config=configuration();config['launch_ready']=True
        config['readiness_sha256']='0'*64
        with patch('examples.resolved_pzt_queue.configuration',return_value=config):
            with self.assertRaisesRegex(ValueError,'readiness evidence changed'):preflight()

    def test_duplicate_queue_refused_before_spawning(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'prepared.json').write_text('{}');(root/'queue-owner.json').write_text('{}')
            with patch('examples.resolved_pzt_queue.ROOT',root), \
                    patch('examples.resolved_pzt_queue.configuration',return_value={'launch_ready':True}), \
                    patch('examples.resolved_pzt_queue.preflight',return_value={}), \
                    patch('examples.resolved_pzt_queue.run_monitored') as run:
                with self.assertRaisesRegex(RuntimeError,'already launched'):run_queue()
            run.assert_not_called()

    def test_identical_recordings_report_zero_and_preserve_unsupported_mode(self):
        t=np.arange(1751)*2e-7
        for f in (50000,250000):
            fast=np.exp(-((t-(48e-6+2.5/f))/8e-6)**2)*np.cos(2*np.pi*f*t)
            slow=np.exp(-((t-(130e-6+2.5/f))/15e-6)**2)*np.cos(2*np.pi*f*t)
            channels=np.column_stack((fast+slow,fast-slow))
            answer=compare(t,channels,channels,f,5000000)
            self.assertEqual(answer['fast_arrival_error_percent'],0.)
            self.assertLess(answer['fast_envelope_shape_error_percent'],1e-10)
            if f==250000:
                self.assertIsNone(answer['slow_arrival_error_percent'])
                self.assertEqual(answer['slow_status'],'above_reported_A0_detection_range')

    def test_serial_queue_saves_and_prints_each_result_before_next_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frozen={'fixture':True}
            (root/'prepared.json').write_text(json.dumps(frozen))
            seen=[]
            def fake_worker(command,*args):
                f=int(command[-2]);directory=Path(command[-1]);directory.mkdir()
                if seen:
                    previous=root/f'{seen[-1]//1000}khz/native'
                    self.assertTrue((previous/'summary.txt').is_file())
                    self.assertIn(f'{seen[-1]//1000} kHz: COMPLETED',(root/'queue.log').read_text())
                seen.append(f)
                comparison={'fast_arrival_error_percent':2.,'fast_status':'exploratory_timing',
                    'slow_arrival_error_percent':None,'slow_status':'unresolved_arrival'}
                result={'completed':True,'wall_seconds':1.,'comparison':comparison}
                for name,key in [('waveforms.npz','waveforms_sha256'),('final-state.npz','final_state_sha256')]:
                    (directory/name).write_bytes(b'unit-test fixture, not simulation data')
                    result[key]=sha256_file(directory/name)
                (directory/'result.json').write_text(json.dumps(result))
                (directory/'comparison.json').write_text(json.dumps(comparison))
                (directory/'summary.txt').write_text('synthetic test fixture')
                return {'exit_code':0}
            with patch('examples.resolved_pzt_queue.ROOT',root), \
                    patch('examples.resolved_pzt_queue.configuration',return_value={'launch_ready':True,'frequencies_hz':[50000,100000]}), \
                    patch('examples.resolved_pzt_queue.preflight',return_value=frozen), \
                    patch('examples.resolved_pzt_queue.run_monitored',side_effect=fake_worker), \
                    patch('sys.stdout',io.StringIO()):
                run_queue()
            self.assertEqual(seen,[50000,100000])
            self.assertEqual(len(json.loads((root/'all-results.json').read_text())),2)
            self.assertIn('Fast wave arrival error: 2.00%',(root/'queue.log').read_text())

    def test_display_records_waiting_status_without_launching(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'prepared.json').write_text(json.dumps({'configuration':{'frequencies_hz':[50000],
                'launch_holds':['mesh refinement']}}))
            screen=io.StringIO();transcript=io.StringIO()
            with patch('examples.resolved_pzt_queue.run_monitored') as launch:
                watch(root,once=True,stream=screen,record_stream=transcript)
            launch.assert_not_called()
            self.assertIn('NOT STARTED',transcript.getvalue())
            self.assertIn('0.0%',transcript.getvalue())
            self.assertIn('test 1/1',transcript.getvalue())
            self.assertEqual(screen.getvalue(),transcript.getvalue())

    def test_zero_signal_is_unresolved_not_perfect_agreement(self):
        t=np.arange(1751)*2e-7;channels=np.zeros((len(t),2))
        answer=compare(t,channels,channels,50000,5000000)
        self.assertIsNone(answer['fast_arrival_error_percent'])
        self.assertIsNone(answer['fast_envelope_shape_error_percent'])
        self.assertEqual(answer['fast_status'],'no_signal')


if __name__=='__main__':unittest.main()
