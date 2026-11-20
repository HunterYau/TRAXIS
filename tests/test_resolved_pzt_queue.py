"""Long-run gate and automatic measured-comparison reporting."""
import unittest
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from examples.resolved_pzt_queue import compare,run_queue,preflight,configuration,PROJECT,queue_resource_limits
from examples.pzt_terminal import watch
from traxis.storage import sha256_file
from traxis.resources import ResourceGuard,ResourceLimitError
from examples.plate_outputs import process_owner, queue_lock


class QueueTests(unittest.TestCase):
    def test_continuation_launches_third_experiment_without_rerunning_first_two(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'prepared.json').write_text('{}')
            selected={**configuration(),'start_experiment':3,'frequencies_hz':[150000,200000,250000]}
            with patch('examples.resolved_pzt_queue.ROOT',root), \
                    patch('examples.resolved_pzt_queue.configuration',return_value=selected), \
                    patch('examples.resolved_pzt_queue.preflight',return_value={}), \
                    patch('examples.resolved_pzt_queue.run_monitored',side_effect=RuntimeError('test launch intercepted')) as run, \
                    patch('sys.stdout',io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'test launch intercepted'):run_queue()
            self.assertEqual(run.call_args.args[0][-2],'150000')
            self.assertEqual(run.call_count,1)
            self.assertIn('Starting experiment 3/5: 150 kHz',(root/'logs/queue.log').read_text())
            self.assertTrue((root/'150khz/INTERRUPTED.txt').is_file())
            self.assertFalse((root/'50khz').exists())
            self.assertFalse((root/'100khz').exists())

    def test_continuation_viewer_shows_original_experiment_number(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'prepared.json').write_text(json.dumps({'configuration':{
                'start_experiment':3,'frequencies_hz':[150000,200000,250000]}}))
            screen=io.StringIO();watch(root,once=True,stream=screen)
            self.assertIn('test 3/5',screen.getvalue())
            self.assertIn('150 kHz',screen.getvalue())

    def test_queue_has_no_live_resource_or_time_stops(self):
        self.assertTrue(all(value is None for value in asdict(queue_resource_limits()).values()))

    def test_queue_reports_high_readings_without_aborting(self):
        guard=ResourceGuard(queue_resource_limits())
        sample={'process_rss_mib':65536,'system_memory_percent':100.,
                'available_mib':0.,'cpu_temperatures_c':[100.]}
        gpu={'used_mib':100,'total_mib':100,'temperature_c':90}
        guard.enforce(sample,[gpu])
        config=configuration()
        self.assertEqual(config['cooling'],{'upper_c':82.,'lower_c':78.})

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
            root=Path(folder);(root/'prepared.json').write_text('{}')
            (root/'queue-owner.json').write_text(json.dumps(process_owner()))
            with patch('examples.resolved_pzt_queue.ROOT',root), \
                    patch('examples.resolved_pzt_queue.configuration',return_value={'launch_ready':True}), \
                    patch('examples.resolved_pzt_queue.preflight',return_value={}), \
                    patch('examples.resolved_pzt_queue.run_monitored') as run:
                with self.assertRaisesRegex(RuntimeError,'already running'):run_queue()
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
                f=int(command[-2]);directory=Path(command[-1])
                if seen:
                    self.assertTrue((root/f'summaries/{seen[-1]//1000}khz.txt').is_file())
                    self.assertIn(f'{seen[-1]//1000} kHz: COMPLETED',(root/'logs/queue.log').read_text())
                    self.assertIn(f'{seen[-1]//1000} kHz: COMPLETED',(root/'all-results.txt').read_text())
                seen.append(f)
                comparison={'fast_arrival_error_percent':2.,'fast_status':'exploratory_timing',
                    'slow_arrival_error_percent':None,'slow_status':'unresolved_arrival'}
                result={'completed':True,'wall_seconds':1.,'comparison':comparison}
                for name,key in [('waveforms.npz','waveforms_sha256'),('final-state.npz','final_state_sha256')]:
                    (directory/name).write_bytes(b'unit-test fixture, not simulation data')
                    result[key]=sha256_file(directory/name)
                (directory/'result.json').write_text(json.dumps(result))
                (directory/'comparison.json').write_text(json.dumps(comparison))
                return {'exit_code':0}
            def monitored(command,*args,**kwargs):
                self.assertIsNone(kwargs['max_log_mib'])
                guard=args[2]
                self.assertFalse(guard.telemetry_errors_fatal)
                self.assertTrue(all(value is None for value in asdict(guard.limits).values()))
                return fake_worker(command,*args)
            with patch('examples.resolved_pzt_queue.ROOT',root), \
                    patch('examples.resolved_pzt_queue.configuration',return_value={'launch_ready':True,'frequencies_hz':[50000,100000],
                        'resource_limits':asdict(queue_resource_limits())}), \
                    patch('examples.resolved_pzt_queue.preflight',return_value=frozen), \
                    patch('examples.resolved_pzt_queue.run_monitored',side_effect=monitored), \
                    patch('sys.stdout',io.StringIO()):
                run_queue()
            self.assertEqual(seen,[50000,100000])
            self.assertEqual(len(json.loads((root/'all-results.json').read_text())),2)
            self.assertIn('S0 wave arrival error: 2.00%',(root/'logs/queue.log').read_text())
            combined=(root/'all-results.txt').read_text()
            self.assertIn('Completed: 2/2',combined)
            self.assertEqual(combined.count('50 kHz: COMPLETED'),1)
            self.assertEqual(combined.count('100 kHz: COMPLETED'),1)

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
