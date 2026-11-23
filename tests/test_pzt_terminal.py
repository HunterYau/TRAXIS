"""Display actual queue states without starting simulations."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.pzt_terminal import progress_state, resource_line, watch,result_lines, export_queue_results


class TextProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'prepared.json').write_text(json.dumps(
            {'configuration': {'frequencies_hz': [50000, 100000]}}))

    def test_partial_log_and_cooling_preserve_progress(self):
        log = self.root/'worker.log'
        log.write_text('warning\n'+json.dumps(dict(phase='propagating', steps_completed=50,
            requested_steps=100, elapsed_seconds=25))+'\n'+
            json.dumps(dict(phase='cooling', steps_completed=50))+'\n{"phase":')
        state = progress_state(log)
        self.assertEqual((state['done'], state['total'], state['phase']), (50, 100, 'cooling'))

    def test_completed_results_print_once_before_next_progress_bar(self):
        first = self.root/'50khz/native'; first.mkdir(parents=True)
        (first/'result.json').write_text(json.dumps(dict(completed=True, wall_seconds=70,
            cpu_force_relative_error=1e-14, modified_energy_relative_drift=1e-16)))
        output = io.StringIO()
        self.assertEqual(watch(self.root, once=True, stream=output), 0)
        text = output.getvalue()
        self.assertEqual(text.count('50 kHz: COMPLETED'), 1)
        self.assertIn('time 00:01:10', text)
        self.assertIn('test 2/2 | 100 kHz', text)
        self.assertIn('pending analysis', text)

    def test_failure_is_reported_without_starting_next_experiment(self):
        first = self.root/'50khz/native'; first.mkdir(parents=True)
        (first/'result.json').write_text(json.dumps(dict(completed=False, error='fixture failed')))
        output = io.StringIO()
        self.assertEqual(watch(self.root, once=True, stream=output), 1)
        self.assertIn('fixture failed', output.getvalue())
        self.assertNotIn('test 2/2', output.getvalue())

    def test_all_saved_experiments_exit_without_waiting(self):
        for frequency in (50, 100):
            folder = self.root/f'{frequency}khz/native'; folder.mkdir(parents=True)
            (folder/'result.json').write_text('{"completed":true}')
        output = io.StringIO()
        with patch('examples.pzt_terminal.time.sleep', side_effect=AssertionError('unexpected wait')):
            self.assertEqual(watch(self.root, stream=output), 0)
        self.assertIn('All 2 experiments completed.', output.getvalue())

    def test_stopped_queue_does_not_look_like_a_running_job(self):
        output = io.StringIO()
        with patch('examples.pzt_terminal.queue_alive', return_value=False):
            self.assertEqual(watch(self.root, stream=output), 2)
        self.assertIn('Queue has stopped', output.getvalue())

    def test_debug_does_not_suppress_later_completion(self):
        (self.root/'prepared.json').write_text(json.dumps(
            {'configuration': {'frequencies_hz': [50000]}}))
        def finish(_):
            directory = self.root/'50khz/native'
            directory.mkdir(parents=True)
            (directory/'result.json').write_text('{"completed":true}')
        output = io.StringIO()
        with patch('examples.pzt_terminal.queue_alive', return_value=True), \
                patch('examples.pzt_terminal.time.sleep', side_effect=finish):
            self.assertEqual(watch(self.root, debug=True, stream=output), 0)
        self.assertIn('Debug log:', output.getvalue())
        self.assertEqual(output.getvalue().count('50 kHz: COMPLETED'), 1)

    def test_unavailable_gpu_does_not_break_display(self):
        with patch('examples.pzt_terminal.shutil.which', return_value=None):
            line = resource_line()
        self.assertIn('RAM', line)
        self.assertIn('GPU readings unavailable', line)

    def test_setup_check_is_not_reported_as_completed_experiment(self):
        lines='\n'.join(result_lines(50000,{'completed':True,'smoke_test':True},self.root))
        self.assertIn('Short setup check: PASSED',lines)
        self.assertNotIn('50 kHz: COMPLETED',lines)

    def test_collective_export_preserves_uncertainty_and_refreshes_without_duplicates(self):
        first = self.root/'50khz/native'; first.mkdir(parents=True)
        saved = json.dumps(dict(completed=True, wall_seconds=70, comparison={
            'fast_status':'exploratory_timing', 'fast_arrival_error_percent':2.,
            'slow_status':'unresolved_arrival', 'slow_arrival_error_percent':None}))
        (first/'result.json').write_text(saved)
        path = export_queue_results(self.root)
        self.assertIn('100 kHz: No saved result yet.', path.read_text())
        second = self.root/'100khz/native'; second.mkdir(parents=True)
        (second/'result.json').write_text(json.dumps(dict(
            completed=False, wall_seconds=5, error='fixture integration failure')))
        export_queue_results(self.root)
        text = path.read_text()
        self.assertEqual(text.count('50 kHz: COMPLETED'), 1)
        self.assertIn('S0 wave arrival error: 2.00%', text)
        self.assertIn('A0 wave arrival error: unresolved (unresolved_arrival)', text)
        self.assertIn('fixture integration failure', text)
        self.assertIn('Completed: 1/2 | failed/incomplete: 1 | awaiting results: 0', text)
        self.assertIn('Total recorded experiment time: 00:01:15', text)
        self.assertNotIn('No saved result yet', text)
        self.assertEqual((first/'result.json').read_text(), saved)

    def test_uncertain_a0_candidate_is_printed_and_exported_with_warning(self):
        for status in ('unresolved_arrival', 'above_reported_A0_detection_range'):
            with self.subTest(status=status):
                trial = self.root/'50khz'
                trial.mkdir(exist_ok=True)
                result = dict(completed=True, comparison={
                    'fast_status':'exploratory_timing', 'fast_arrival_error_percent':2.,
                    'slow_status':status, 'slow_arrival_error_percent':None,
                    'slow_details':{'raw_candidate_error_percent':3.842049}})
                saved = json.dumps(result)
                (trial/'result.json').write_text(saved)
                expected = ('A0 wave arrival error: 3.84% (candidate only; may be inaccurate: '
                            'A0 could not be identified reliably; '+status+')')
                self.assertIn(expected, '\n'.join(result_lines(50000, result, trial)))
                self.assertIn(expected, export_queue_results(self.root).read_text())
                self.assertEqual((trial/'result.json').read_text(), saved)
                result['comparison']['slow_status']='no_signal'
                self.assertIn('A0 wave arrival error: unresolved (no_signal)',
                              '\n'.join(result_lines(50000, result, trial)))

    def test_collective_export_reports_worker_exit_without_saved_result(self):
        trial = self.root/'50khz'; trial.mkdir()
        (trial/'launcher.json').write_text('{"exit_code":1}')
        text = export_queue_results(self.root).read_text()
        self.assertIn('50 kHz: FAILED | worker exit code 1', text)
        self.assertIn('Completed: 0/2 | failed/incomplete: 1 | awaiting results: 1', text)

    def test_gpu_readings_are_labeled_machine_wide(self):
        import subprocess
        result = subprocess.CompletedProcess([], 0, stdout='95, 6000, 8192, 70\n')
        with patch('examples.pzt_terminal.shutil.which', return_value='nvidia-smi'), \
                patch('examples.pzt_terminal.subprocess.run', return_value=result):
            line = resource_line()
        self.assertIn('Machine:', line)
        self.assertIn('95% busy', line)
        self.assertIn('6000/8192 MiB', line)
        self.assertIn('shared GPU memory not measured', line)


if __name__ == '__main__':
    unittest.main()
