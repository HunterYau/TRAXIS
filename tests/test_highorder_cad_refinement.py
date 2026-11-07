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
            'float64_traces_sha256': 'fixture',
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
                       {'max_total_solves': 11}, {'reference': HighOrderWaveSettings()}, {'compare_scattering': 1}):
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

    def test_direct_arrival_agreement_cannot_hide_scattering_failure(self):
        def paired(scenario, settings, output, guard, plan, cache, runtime):
            _, row, runtime = self.produce(scenario, settings, output, guard, plan, cache, runtime)
            phase = 2*np.pi*100000*np.arange(scenario.acquisition.samples)*scenario.acquisition.sample_dt_s
            values = 1e-9*np.sin(phase)
            if plan['state'] == 'current':
                values += 1e-11*((1+5*(settings.mesh_size_m/.002)**2)*np.sin(phase)
                    + .1/settings.substeps_per_output**2*np.cos(phase))
            return np.tile(values, (len(plan['actions']), 1)), row, runtime
        settings = HighOrderCADRefinementSettings(compare_scattering=True, max_total_solves=10)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'study'
            with patch('traxis.highorder_cad_refinement._checked_case', side_effect=paired) as solve:
                report = run_highorder_cad_refinement(local_scenario(), settings, output)
            self.assertEqual(solve.call_count, 10)
            self.assertTrue(report['current_spatial_screen_passed'])
            self.assertFalse(report['scattered_spatial_screen_passed'])
            self.assertFalse(report['discretization_screen_passed'])
            self.assertTrue(report['scattered_temporal_screen_passed'])
            plan = json.loads((output/'refinement-plan.json').read_text())
            self.assertEqual(plan['requested_forward_solves'], 10)
            view = json.loads((output/'scattered/refinement-report.json').read_text())
            self.assertFalse(view['discretization_screen_passed'])
            self.assertFalse(view['ready_for_training_data'])
            with np.load(output/'scattered/case-02/solver-traces.npz') as saved:
                phase = 2*np.pi*100000*np.arange(64)*local_scenario().acquisition.sample_dt_s
                baseline = 1e-9*np.sin(phase)
                signal = 1e-11*(6*np.sin(phase)+.1/16**2*np.cos(phase))
                # Subtraction retains float64 cancellation; no signal fit or
                # normalization is permitted between paired states.
                expected = (baseline+signal)-baseline
                np.testing.assert_array_equal(saved['displacement_m'][0], expected)

    def test_scattering_requires_a_recess_no_coupling_drift_and_ten_solve_budget(self):
        scenario = local_scenario()
        settings = HighOrderCADRefinementSettings(compare_scattering=True, max_total_solves=10)
        drifted = replace(scenario, nuisance=replace(scenario.nuisance, current_gain_drift=(1.1,)*8))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'absent'
            for case, config, state in ((local_scenario(intact=True), settings, 'current'),
                                       (scenario, settings, 'intact'), (drifted, settings, 'current'),
                                       (scenario, replace(settings, max_total_solves=5), 'current')):
                with self.assertRaises(ValueError):
                    run_highorder_cad_refinement(case, config, output, state=state)
            self.assertFalse(output.exists())

    def test_changed_intact_mesh_cannot_pass_a_paired_time_comparison(self):
        def changed(*args):
            values, row, runtime = self.produce(*args)
            if args[4]['state'] == 'intact' and args[2].name == 'case-04':
                row['geometry_load_fingerprint'] = 'changed-intact-mesh'
            return values, row, runtime
        settings = HighOrderCADRefinementSettings(compare_scattering=True, max_total_solves=10)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'study'
            with patch('traxis.highorder_cad_refinement._checked_case', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'scattered time refinement'):
                    run_highorder_cad_refinement(local_scenario(), settings, output)
            report = json.loads((output/'refinement-report.json').read_text())
            self.assertFalse(report['completed'])
            self.assertFalse(report['discretization_screen_passed'])
