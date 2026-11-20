"""Filesystem lifecycle checks with synthetic workers; no GPU simulation."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples import resolved_pzt_queue as queue
from examples.plate_outputs import queue_lock, process_owner, trial_status
from examples.pzt_terminal import export_queue_results, watch
from examples import organize_plate_outputs as organizer
from traxis.storage import sha256_file


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {**queue.configuration(), 'launch_ready':True, 'frequencies_hz':[50000,100000]}
        self.frozen = {'configuration':self.config}
        (self.root/'prepared.json').write_text(json.dumps(self.frozen))
        self.calls = []
        self.fail_second = True

    def worker(self, command, *args, **kwargs):
        frequency, trial = int(command[-2]), Path(command[-1])
        self.calls.append(frequency)
        (trial/'worker.log').write_text('synthetic progress\n')
        if frequency == 100000 and self.fail_second:
            (trial/'partial-data.txt').write_text('unfinished output retained')
            return {'exit_code':1}
        comparison = {'fast_status':'exploratory_timing', 'fast_arrival_error_percent':2.,
                      'slow_status':'unresolved_arrival', 'slow_arrival_error_percent':None}
        result = {'completed':True, 'wall_seconds':1., 'comparison':comparison}
        for name,key in [('waveforms.npz','waveforms_sha256'),('final-state.npz','final_state_sha256')]:
            (trial/name).write_bytes(b'fixture output, not real simulation')
            result[key] = sha256_file(trial/name)
        (trial/'comparison.json').write_text(json.dumps(comparison))
        (trial/'result.json').write_text(json.dumps(result))
        return {'exit_code':0}

    def run_saved(self):
        with patch.object(queue,'ROOT',self.root), \
                patch.object(queue,'configuration',return_value=self.config), \
                patch.object(queue,'preflight',return_value=self.frozen), \
                patch.object(queue,'run_monitored',side_effect=self.worker), \
                patch('sys.stdout',io.StringIO()):
            queue.run_queue()

    def test_retry_preserves_failed_attempt_and_skips_completed_frequency(self):
        with self.assertRaisesRegex(RuntimeError,'100000 Hz failed'):
            self.run_saved()
        first_bytes = (self.root/'50khz/result.json').read_bytes()
        self.assertTrue((self.root/'100khz/FAILED.txt').is_file())
        self.assertIn('failed/incomplete: 1',(self.root/'all-results.txt').read_text())
        self.fail_second = False
        self.run_saved()
        self.assertEqual(self.calls,[50000,100000,100000])
        self.assertEqual((self.root/'50khz/result.json').read_bytes(),first_bytes)
        archive = self.root/'100khz/previous-attempts/attempt-001-unfinished'
        self.assertEqual((archive/'partial-data.txt').read_text(),'unfinished output retained')
        self.assertTrue((archive/'FAILED.txt').is_file())
        self.assertFalse((self.root/'100khz/FAILED.txt').exists())
        self.assertIn('Completed: 2/2',(self.root/'all-results.txt').read_text())
        self.assertTrue((self.root/'summaries/50khz.txt').is_file())
        self.assertTrue((self.root/'summaries/100khz.txt').is_file())
        self.run_saved()
        self.assertEqual(self.calls,[50000,100000,100000])
        self.assertEqual(len(json.loads((self.root/'all-results.json').read_text())),2)

    def test_abrupt_stop_detected_when_export_or_display_reopens(self):
        trial = self.root/'50khz'; trial.mkdir()
        trial_status(trial,'running')
        text = export_queue_results(self.root).read_text()
        self.assertIn('50 kHz: INTERRUPTED / UNFINISHED',text)
        self.assertTrue((trial/'INTERRUPTED.txt').is_file())
        screen = io.StringIO()
        self.assertEqual(watch(self.root,once=True,stream=screen),2)
        self.assertIn('INTERRUPTED / UNFINISHED',screen.getvalue())

    def test_source_change_refused_before_archiving_partial_data(self):
        trial = self.root/'50khz'; trial.mkdir()
        (trial/'partial-data.txt').write_text('keep me')
        with patch.object(queue,'ROOT',self.root), \
                patch.object(queue,'configuration',return_value=self.config), \
                patch.object(queue,'preflight',return_value={'different':True}), \
                patch.object(queue,'run_monitored') as worker:
            with self.assertRaisesRegex(ValueError,'changed'):
                queue.run_queue()
        worker.assert_not_called()
        self.assertEqual((trial/'partial-data.txt').read_text(),'keep me')
        self.assertFalse((trial/'previous-attempts').exists())

    def test_live_orphan_worker_refuses_duplicate_resume(self):
        trial = self.root/'50khz'; trial.mkdir()
        (trial/'worker-owner.json').write_text(json.dumps(process_owner()))
        with patch.object(queue,'ROOT',self.root), \
                patch.object(queue,'configuration',return_value=self.config), \
                patch.object(queue,'run_monitored') as worker:
            with self.assertRaisesRegex(RuntimeError,'worker is still running'):
                queue.run_queue()
        worker.assert_not_called()

    def test_os_lock_blocks_another_scheduler_and_releases_after_exit(self):
        with queue_lock(self.root):
            with self.assertRaisesRegex(RuntimeError,'already running'):
                with queue_lock(self.root):
                    self.fail('second scheduler acquired an active lock')
        with queue_lock(self.root):
            pass

    def test_progress_display_follows_failed_attempt_to_successful_retry(self):
        trial = self.root/'50khz'; trial.mkdir()
        (trial/'result.json').write_text('{"completed":false,"error":"fixture stopped"}')
        def finish(_):
            for frequency in (50,100):
                folder = self.root/f'{frequency}khz'; folder.mkdir(exist_ok=True)
                (folder/'result.json').write_text('{"completed":true}')
        screen = io.StringIO()
        with patch('examples.pzt_terminal.time.sleep',side_effect=finish):
            self.assertEqual(watch(self.root,stream=screen,follow_retries=True),0)
        self.assertIn('50 kHz: FAILED / INCOMPLETE',screen.getvalue())
        self.assertIn('50 kHz: COMPLETED',screen.getvalue())
        self.assertIn('All 2 experiments completed.',screen.getvalue())

    def test_migration_preserves_original_records_and_is_repeatable(self):
        project = self.root/'project'
        source = project/'outputs/old-run'
        trial = source/'50khz/native'; trial.mkdir(parents=True)
        saved = '{"completed":true,"wall_seconds":2}'
        (trial/'result.json').write_text(saved)
        (trial/'waveforms.npz').write_bytes(b'fixture waveform')
        (trial/'summary.txt').write_text('original summary')
        (source/'prepared.json').write_text(json.dumps({'configuration':{'frequencies_hz':[50000]}}))
        (source/'queue.log').write_text('original log')
        runs = project/'outputs/simulation-runs'
        target = runs/'current-run'
        with patch.object(organizer,'PROJECT',project), \
                patch.object(organizer,'RUNS_ROOT',runs), \
                patch.object(organizer,'CURRENT_RUN',target), \
                patch.object(organizer,'LEGACY_RUNS',{'old-run':'current-run'}):
            organizer.migrate()
            organizer.migrate()
        self.assertFalse(source.exists())
        self.assertEqual((target/'50khz/result.json').read_text(),saved)
        self.assertEqual((target/'50khz/waveforms.npz').read_bytes(),b'fixture waveform')
        self.assertEqual((target/'provenance/original-summaries/50khz.txt').read_text(),'original summary')
        self.assertEqual((target/'logs/queue.log').read_text(),'original log')
        self.assertTrue((target/'all-results.txt').is_file())
        self.assertTrue((target/'summaries/50khz.txt').is_file())


if __name__=='__main__':
    unittest.main()
