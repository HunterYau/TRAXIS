import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from examples.matrixfree_capacity import measure_capacity
from examples.matrixfree_step import capacity_context, checked_settings, measure_step
from tests.test_matrixfree import capacity_settings
from traxis.config import Plate
from traxis.resources import ResourceGuard, ResourceLimits


def step_settings():
    return dict(dt_s=1.25e-7,frequency_hz=100000.,cycles=2,peak_patch_force_n=1.,center_xy_m=[.01,.008],
        radius_m=.003,verify_sparse=True,iterative=dict(relative_tolerance=1e-11,residual_limit=1e-9,
        max_iterations=500,max_dofs=20000,max_workspace_mib=64))


class FirstStepPreflightTests(unittest.TestCase):
    def test_settings_refuse_unresolved_drive_and_outside_or_zero_source(self):
        plate = Plate(.04,.03,.003)
        checked_settings(step_settings(),plate)
        for changes in ({'dt_s':1e-5},{'center_xy_m':[0.,0.]},{'cycles':0},{'radius_m':0},
                        {'peak_patch_force_n':float('nan')},{'verify_sparse':1},{'unexpected':0},
                        {'backend':'unknown'},{'backend':'native','gpu':{}},{'backend':'gpu_elements'},
                        {'preconditioner':'element_schwarz'},{'preconditioner':'unknown'}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    checked_settings({**step_settings(),**changes},plate)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                capacity_context(Path(directory))


@unittest.skipUnless(os.environ.get('TRAXIS_HIGHORDER_CHECKS') == '1' and importlib.util.find_spec('ngsolve'),
                     'enable optional isolated native first-step checks')
class NativeFirstStepTests(unittest.TestCase):
    def test_actual_load_and_first_step_match_direct_sparse_newmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small = {**capacity_settings(),'mesh_size_m':.008}
            capacity = measure_capacity(small,root/'capacity',
                ResourceGuard(ResourceLimits(max_seconds=45,max_rss_mib=1024)))
            self.assertTrue(capacity['completed'],capacity.get('error'))
            result = measure_step(root/'capacity',step_settings(),root/'step',
                ResourceGuard(ResourceLimits(max_seconds=45,max_rss_mib=1024)))
            log = root/'step/worker.log'
            self.assertTrue(result['completed'],str(result.get('error'))+'\n'+(log.read_text()[-3000:] if log.exists() else ''))
            self.assertEqual(result['steps_completed'],1)
            self.assertFalse(result['ready_for_training_data'])
            native = json.loads((root/'step/native/step-report.json').read_text())
            self.assertTrue(native['sparse_reference_assembled'])
            self.assertLess(native['direct_acceleration_relative_l2'],1e-7)
            self.assertLess(native['direct_probe_relative_error'],1e-7)
            self.assertLess(native['relative_energy_balance_error'],1e-8)
            self.assertEqual(native['solve_progress'][-1]['phase'],'solve_end')
            self.assertLess(native['solve_progress'][-1]['true_relative_residual'],1e-9)
            bad = json.loads((root/'capacity/monitor-result.json').read_text())
            bad['completed'] = False
            (root/'capacity/monitor-result.json').write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                capacity_context(root/'capacity')
