from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_forward import local_scenario
from traxis.highorder_cad_refinement import HighOrderCADRefinementSettings, run_highorder_cad_refinement
from traxis.highorder_waveforms import HighOrderWaveSettings
from traxis.storage import canonical_hash


class CADRefinementTests(unittest.TestCase):
    def produce(self, scenario, settings, output, guard, plan, cache, runtime):
        t = np.arange(scenario.acquisition.samples)*scenario.acquisition.sample_dt_s
        phase = 2*np.pi*100000*t
        values = 1e-9*((1+.1*(settings.mesh_size_m/.002)**2)*np.sin(phase)
            + .1/settings.substeps_per_output**2*np.cos(phase))
        row = {'directory': output.name, 'mesh_size_m': settings.mesh_size_m,
            'substeps_per_output': settings.substeps_per_output,
            'dt_internal_s': scenario.acquisition.sample_dt_s/settings.substeps_per_output,
            'linear_mesh': {'maximum_edge_m': settings.mesh_size_m, 'maximum_element_z_span_m': .001},
            'geometry_load_fingerprint': canonical_hash({'mesh_size_m': settings.mesh_size_m})}
        return np.tile(values, (len(plan['actions']), 1)), row, {'fixture': True}

    def test_frozen_five_case_axes_and_failed_spatial_screen_are_retained(self):
        scenario, settings = local_scenario(), HighOrderCADRefinementSettings()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'study'
            with patch('traxis.highorder_cad_refinement._checked_case', side_effect=self.produce) as solve:
                report = run_highorder_cad_refinement(scenario, settings, output)
            self.assertEqual([(c.args[1].mesh_size_m, c.args[1].substeps_per_output) for c in solve.call_args_list],
                [(.003,16), (.0025,16), (.002,16), (.002,4), (.002,8)])
            self.assertTrue(report['completed'])
            self.assertFalse(report['spatial_screen_passed'])
            self.assertTrue(report['temporal_screen_passed'])
            self.assertFalse(report['discretization_screen_passed'])
            self.assertAlmostEqual(report['observed_temporal_order'], 2., places=8)
            self.assertFalse(report['ready_for_training_data'])
            plan = json.loads((output/'refinement-plan.json').read_text())
            self.assertEqual(plan['state'], 'current')
            self.assertEqual(len(plan['scenario']['recesses']), 1)
            for call in solve.call_args_list:
                self.assertEqual(call.args[4]['fingerprint'], plan['fingerprint'])

    def test_temporal_geometry_mismatch_cannot_finish_or_pass(self):
        def changed(*args):
            values, row, runtime = self.produce(*args)
            if args[2].name == 'case-04':
                row['geometry_load_fingerprint'] = 'changed'
            return values, row, runtime
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'study'
            with patch('traxis.highorder_cad_refinement._checked_case', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'spatial mesh'):
                    run_highorder_cad_refinement(local_scenario(), HighOrderCADRefinementSettings(), output)
            report = json.loads((output/'refinement-report.json').read_text())
            self.assertFalse(report['completed'])
            self.assertFalse(report['discretization_screen_passed'])

    def test_invalid_settings_budgets_and_noise_fail_before_native_work(self):
        for values in ({'mesh_sizes_m': (.003,.003,.002)}, {'substeps_per_output': (4,8,17)},
                       {'max_total_solves': 6}, {'reference': HighOrderWaveSettings()}):
            with self.assertRaises(ValueError):
                HighOrderCADRefinementSettings(**values)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'absent'
            scenario = local_scenario()
            with self.assertRaises(ValueError):
                run_highorder_cad_refinement(scenario, HighOrderCADRefinementSettings(max_total_solves=4), output)
            noisy = replace(scenario, nuisance=replace(scenario.nuisance, relative_noise_std=.01))
            with self.assertRaises(ValueError):
                run_highorder_cad_refinement(noisy, HighOrderCADRefinementSettings(), output)
            self.assertFalse(output.exists())
