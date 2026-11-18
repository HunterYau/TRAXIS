"""Display actual queue states without starting simulations."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.pzt_terminal import progress_state, resource_line, watch


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
        self.assertNotIn('100 kHz', output.getvalue())

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
