import importlib.util
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from traxis.config import Plate
from traxis.highorder_patch_verification import run_native_job, verify_highorder_patches
from traxis.highorder_patches import highorder_surface_patch
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import canonical_hash, write_json


class HighOrderPatchPreflightTests(unittest.TestCase):
    def test_larger_footprints_need_explicit_limits_and_screen_before_allocation(self):
        from types import SimpleNamespace
        system = SimpleNamespace(scalar_space=SimpleNamespace(globalorder=3,ndof=100000),
                                 mesh=SimpleNamespace(ne=100000,nv=20000))
        with self.assertRaisesRegex(ValueError,'bounded'):
            highorder_surface_patch(system,Plate(),(.02,.02),.003)
        with self.assertRaisesRegex(ValueError,'element/vertex'):
            highorder_surface_patch(system,Plate(),(.02,.02),.003,max_scalar_dofs=100000)
        with self.assertRaisesRegex(MemoryError,'workspace'):
            highorder_surface_patch(system,Plate(),(.02,.02),.003,max_scalar_dofs=100000,
                                    max_tetrahedra=100000,max_workspace_mib=1e-6)

    def test_command_distinguishes_operational_and_numerical_failure(self):
        from traxis.cli import main
        args = ['traxis', 'verify-highorder-patches', '--elmer-home', 'unused', '--output', 'unused']
        for error in (ModuleNotFoundError('optional solver missing'), ArithmeticError('native check failed')):
            with patch('sys.argv', args), redirect_stderr(io.StringIO()) as stderr:
                with patch('traxis.highorder_patch_verification.verify_highorder_patches', side_effect=error):
                    with self.assertRaises(SystemExit) as stopped:
                        main()
                self.assertEqual(stopped.exception.code, 1)
                self.assertIn(str(error), stderr.getvalue())
        with patch('sys.argv', args), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
            with patch('traxis.highorder_patch_verification.verify_highorder_patches', return_value={'completed': True, 'passed': False}):
                with self.assertRaises(SystemExit) as stopped:
                    main()
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn('screen failed', stderr.getvalue())

    def test_invalid_patch_parameters_fail_before_accessing_native_space(self):
        for center, radius, direction in (((0, 0), 0., (0, 0, 1)), ((float('nan'), 0), .003, (0, 0, 1)),
                                          ((0, 0), .003, (0, 0, 0)), ((0, 0), .003, (1j, 0, 0))):
            with self.assertRaises(ValueError):
                highorder_surface_patch(None, Plate(), center, radius, direction)

    def test_missing_bindings_and_tampered_jobs_fail_without_native_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = {'input_sha256': {}}
            job['fingerprint'] = canonical_hash(job)
            write_json(root / 'job.json', job)
            with self.assertRaisesRegex(ValueError, 'bind'):
                run_native_job(root / 'job.json', root / 'native', ResourceGuard())
            job['input_sha256'] = {'changed': 'value'}
            write_json(root / 'job.json', job, overwrite=True)
            with self.assertRaisesRegex(ValueError, 'changed'):
                run_native_job(root / 'job.json', root / 'native', ResourceGuard())
            self.assertFalse((root / 'native').exists())

    def test_changed_archive_is_rejected_before_native_runtime_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source-snapshot.zip').write_bytes(b'changed')
            (root / 'highorder-aperture-plan.json').write_text('{}')
            job = {'input_sha256': {'highorder-aperture-plan.json': 'bad', 'source-snapshot.zip': 'bad'}}
            job['fingerprint'] = canonical_hash(job)
            write_json(root / 'job.json', job)
            with self.assertRaisesRegex(ValueError, 'input changed'):
                run_native_job(root / 'job.json', root / 'native', ResourceGuard())
            self.assertFalse((root / 'native').exists())


@unittest.skipUnless(os.environ.get('TRAXIS_HIGHORDER_CHECKS') == '1' and os.environ.get('TRAXIS_ELMER_HOME')
                     and importlib.util.find_spec('ngsolve'), 'enable optional isolated NGSolve/Elmer aperture checks')
class NativeHighOrderPatchTests(unittest.TestCase):
    def test_full_half_and_quarter_disks_and_independent_native_transients(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'reference'
            result = verify_highorder_patches(Path(os.environ['TRAXIS_ELMER_HOME']), output,
                ResourceGuard(ResourceLimits(max_seconds=60, max_rss_mib=1024)))
            self.assertTrue(result['completed'])
            self.assertTrue(result['passed'])
            self.assertFalse(result['ready_for_training_data'])
            self.assertEqual(len(result['cases']), 12)
            self.assertEqual([row['order'] for row in result['waves']], [1, 2, 3, 4])
            for case in result['cases']:
                self.assertTrue(case['passed'])
                self.assertEqual(len(case['patches']), 4)
                for patch in case['patches']:
                    self.assertTrue(patch['passed'])
                    self.assertLess(patch['moments']['maximum_absolute_error'], 1e-9)
                    if case['order'] == 2 and case['mesh'] != 'refined':
                        self.assertLess(patch['p2_covector_relative_error'], 1e-10)
            for wave in result['waves']:
                self.assertTrue(wave['passed'])
                self.assertLess(wave['reciprocity_relative_error'], 1e-8)
                if wave['order'] == 2:
                    self.assertLess(max(wave['native_elmer_trace_relative_errors']), 1e-8)
            report = json.loads((output / 'highorder-aperture-report.json').read_text())
            self.assertEqual(report['fingerprint'], canonical_hash({k: v for k, v in report.items() if k != 'fingerprint'}))
