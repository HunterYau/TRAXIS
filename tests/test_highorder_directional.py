from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_forward import local_scenario
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.highorder_directional import HighOrderDirectionalSettings, run_highorder_directional
from traxis.highorder_waveforms import HighOrderWaveSettings, reference_actions
from traxis.storage import canonical_hash, sha256_file, write_json
from traxis.waveform_checks import plot_waveform_sweep


class HighOrderDirectionalTests(unittest.TestCase):
    def producer(self, scenario, settings, output, guard, **kwargs):
        output.mkdir()
        actions = reference_actions(scenario, settings, kwargs['transmitter'], kwargs['frequency_hz'])
        plan = {'runtime': {'fixture': True}, 'source_sha256': reference_source_hashes(), 'scenario': scenario.to_dict(),
            'settings': asdict(settings), 'actions': [asdict(a) for a in actions], 'state': 'intact'}
        plan['fingerprint'] = canonical_hash(plan)
        write_json(output / 'case-plan.json', plan)
        times = np.arange(scenario.acquisition.samples)*scenario.acquisition.sample_dt_s
        phase = 2*np.pi*100000*times
        nx, _, nz = settings.cells
        trace = 1e-9*((1+100/nx**2)*np.sin(phase)+.02/nz**2*np.cos(phase)+.1/settings.substeps_per_output**2*np.sin(2*phase))
        values = np.tile(trace, (len(actions), 1))
        np.savez_compressed(output / 'solver-traces.npz', time_s=times, displacement_m=values)
        geometry = {'mesh': str(settings.cells), 'loads': 'fixed physical footprints', 'order': settings.order}
        measured = {'completed': True, 'plan_fingerprint': plan['fingerprint'], 'order': settings.order, 'cells': settings.cells,
            'dofs': 100, 'dt_internal_s': scenario.acquisition.sample_dt_s/settings.substeps_per_output,
            'internal_steps': (scenario.acquisition.samples-1)*settings.substeps_per_output, 'cache_hit': False, 'forward_solves': 1,
            'linear_mesh': {'maximum_edge_m': float(np.sqrt(2*(.04/nx)**2+(.003/nz)**2)), 'maximum_element_z_span_m': .003/nz},
            'geometry_load_sha256': geometry, 'geometry_load_fingerprint': canonical_hash(geometry), 'native_directory': 'fixture',
            'worker_report_sha256': 'fixture', 'stiffness_nonzeros': 1, 'mass_nonzeros': 1, 'lu_factor_nonzeros': 1,
            'maximum_relative_linear_residual': 0., 'postburst_relative_energy_range': 0., 'wall_seconds': 0.,
            'trace_sha256': sha256_file(output / 'solver-traces.npz')}
        write_json(output / 'waveform-case.json', measured)
        return values, measured

    def test_separate_axes_checked_arrays_and_non_geometric_thickness_levels(self):
        settings = HighOrderDirectionalSettings()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'study'
            with patch('traxis.highorder_directional.run_highorder_waveform_case', side_effect=self.producer) as solve:
                report = run_highorder_directional(local_scenario(intact=True), settings, output)
            self.assertEqual([(call.args[1].cells, call.args[1].substeps_per_output) for call in solve.call_args_list],
                [((3,3,3),16), ((4,4,3),16), ((5,5,3),16), ((5,5,1),16), ((5,5,2),16), ((5,5,3),4), ((5,5,3),8)])
            self.assertTrue(report['completed'])
            self.assertFalse(report['in_plane_screen_passed'])
            self.assertTrue(report['thickness_screen_passed'])
            self.assertTrue(report['temporal_screen_passed'])
            self.assertFalse(report['ready_for_training_data'])
            self.assertAlmostEqual(report['observed_temporal_order'], 2., places=8)
            self.assertIsNone(report['observed_thickness_order'])
            if importlib.util.find_spec('matplotlib'):
                figures = plot_waveform_sweep(output, Path(directory) / 'figures', (0,), directional=True)
                self.assertEqual(len(figures['files']), 4)
                path = output / 'case-00/solver-traces.npz'
                path.write_bytes(path.read_bytes()+b'changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    plot_waveform_sweep(output, Path(directory) / 'bad-figures', (0,), directional=True)

    def test_invalid_budgets_and_axes_fail_before_native_work(self):
        scenario = local_scenario(intact=True)
        for values in ({'in_plane_cells': (3,3,5)}, {'thickness_cells': (1,0,3)}, {'substeps_per_output': (4,8,17)}, {'max_total_solves': 8}):
            with self.assertRaises(ValueError):
                HighOrderDirectionalSettings(**values)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'absent'
            for settings in (HighOrderDirectionalSettings(max_total_solves=6),
                HighOrderDirectionalSettings(reference=HighOrderWaveSettings(max_internal_steps=1)),
                HighOrderDirectionalSettings(in_plane_cells=(3,100,1000))):
                with self.assertRaises((ValueError, MemoryError)):
                    run_highorder_directional(scenario, settings, output)
            self.assertFalse(output.exists())

    def test_changed_temporal_operators_and_returned_arrays_cannot_pass(self):
        for corrupt in ('geometry', 'arrays'):
            def changed(*args, **kwargs):
                values, measured = self.producer(*args, **kwargs)
                if corrupt == 'geometry' and args[2].name == 'case-06':
                    measured['geometry_load_fingerprint'] = 'changed'
                if corrupt == 'arrays':
                    values = values+1e-9
                return values, measured
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'study'
                with patch('traxis.highorder_directional.run_highorder_waveform_case', side_effect=changed):
                    with self.assertRaises(ValueError):
                        run_highorder_directional(local_scenario(intact=True), HighOrderDirectionalSettings(), output)
                report = json.loads((output / 'directional-report.json').read_text())
                self.assertFalse(report['completed'])
                self.assertFalse(report['discretization_screen_passed'])
