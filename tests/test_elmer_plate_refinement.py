from dataclasses import asdict, replace
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_forward import local_scenario
from traxis.acquisition import action_catalog
from traxis.elmer_plate_refinement import ElmerPlateRefinementSettings, run_elmer_plate_refinement
from traxis.elmer_waveforms import ElmerWaveSettings
from traxis.resources import ResourceLimitError
from traxis.storage import canonical_hash, sha256_file, write_json
from traxis.waveform_checks import plot_waveform_sweep


class PlateRefinementTests(unittest.TestCase):
    def producer(self, scenario, settings, home, output, guard, **kwargs):
        """Synthetic nonzero traces isolate study scheduling and comparisons."""
        output.mkdir()
        (output / 'mesh').mkdir()
        (output / 'mesh/mesh.nodes').write_text(str(settings.mesh_size_m))
        (output / 'loads-probes.npz').write_bytes(b'fixed test aperture')
        actions = [asdict(a) for a in action_catalog(8, (100000.,)) if a.transmitter == kwargs['transmitter']]
        source = {p.name: sha256_file(p) for p in sorted(Path('traxis').glob('*.py'))}
        case_plan = {'source_sha256': source, 'kernel_sha256': {'solver': 'test'}, 'scenario': asdict(scenario),
                     'actions': actions, 'state': kwargs['state'], 'settings': asdict(settings), 'runtime': {'test': True}}
        case_plan['fingerprint'] = canonical_hash(case_plan)
        write_json(output / 'case-plan.json', case_plan)
        times = np.arange(scenario.acquisition.samples) * scenario.acquisition.sample_dt_s
        phase = 2 * np.pi * 100000 * times
        trace = 1e-9 * ((1 + (settings.mesh_size_m / .003)**2) * np.sin(phase) + .1 / settings.substeps_per_output**2 * np.cos(phase))
        values = np.tile(trace, (len(actions), 1))
        np.savez_compressed(output / 'solver-traces.npz', time_s=times, displacement_m=values)
        measured = {'completed': True, 'dt_internal_s': scenario.acquisition.sample_dt_s / settings.substeps_per_output,
                    'internal_steps': (scenario.acquisition.samples - 1) * settings.substeps_per_output,
                    'quadratic_nodes': 100, 'tetrahedra': 100, 'forward_solves': 1, 'cache_hit': False,
                    'wall_seconds': 0., 'computation': {}, 'trace_sha256': sha256_file(output / 'solver-traces.npz'),
                    'linear_mesh': {'maximum_edge_m': .005, 'maximum_element_z_span_m': .003}}
        write_json(output / 'waveform-case.json', measured)
        return values, measured

    def test_independent_axes_raw_error_and_second_order_time_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'study'
            with patch('traxis.elmer_plate_refinement.run_elmer_waveform_case', side_effect=self.producer) as solve:
                report = run_elmer_plate_refinement(local_scenario(intact=True), ElmerPlateRefinementSettings(), Path('unused'), output)
            self.assertTrue(report['completed'])
            self.assertFalse(report['spatial_screen_passed'])
            self.assertTrue(report['temporal_screen_passed'])
            self.assertFalse(report['ready_for_training_data'])
            self.assertAlmostEqual(report['observed_temporal_order'], 2., places=10)
            definitions = [(call.args[1].mesh_size_m, call.args[1].substeps_per_output) for call in solve.call_args_list]
            self.assertEqual(definitions, [(.003, 16), (.002, 16), (.0015, 16), (.0015, 4), (.0015, 8)])
            self.assertEqual(len({report['cases'][i]['geometry_load_fingerprint'] for i in (2, 3, 4)}), 1)
            if importlib.util.find_spec('matplotlib'):
                result = plot_waveform_sweep(output, Path(directory) / 'figures', records=(0,), reference=True)
                self.assertEqual(len(result['files']), 2)
                path = output / 'case-00/solver-traces.npz'
                path.write_bytes(path.read_bytes() + b'changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    plot_waveform_sweep(output, Path(directory) / 'bad-figures', records=(0,), reference=True)

    def test_preflight_rejects_invalid_axes_noise_and_solve_step_budgets(self):
        for values in ({'mesh_sizes_m': (.003, .004, .001)}, {'substeps_per_output': (4, 8, 15)}):
            with self.assertRaises(ValueError):
                ElmerPlateRefinementSettings(**values)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'must-not-exist'
            scenario = local_scenario(intact=True)
            for settings, pattern in ((ElmerPlateRefinementSettings(max_total_solves=4), 'solve budget'),
                (ElmerPlateRefinementSettings(reference=ElmerWaveSettings(max_internal_steps=1)), 'internal-step')):
                with self.assertRaisesRegex(ValueError, pattern):
                    run_elmer_plate_refinement(scenario, settings, Path('absent'), output)
            noisy = replace(scenario, nuisance=replace(scenario.nuisance, relative_noise_std=.01))
            with self.assertRaisesRegex(ValueError, 'zero stochastic noise'):
                run_elmer_plate_refinement(noisy, ElmerPlateRefinementSettings(), Path('absent'), output)
            self.assertFalse(output.exists())

    def test_temporal_mesh_changes_are_rejected_and_partial_report_is_retained(self):
        def changed(*args, **kwargs):
            result = self.producer(*args, **kwargs)
            if args[3].name == 'case-04':
                (args[3] / 'mesh/mesh.nodes').write_text('changed spatial mesh')
            return result
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'study'
            with patch('traxis.elmer_plate_refinement.run_elmer_waveform_case', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'spatial mesh'):
                    run_elmer_plate_refinement(local_scenario(intact=True), ElmerPlateRefinementSettings(), Path('unused'), output)
            saved = json.loads((output / 'refinement-report.json').read_text())
            self.assertFalse(saved['completed'])
            self.assertEqual(len(saved['cases']), 5)
            self.assertEqual(canonical_hash({k: v for k, v in saved.items() if k != 'fingerprint'}), saved['fingerprint'])

    def test_interruption_preserves_completed_cases_without_passing_screen(self):
        count = 0
        def interrupted(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 3:
                raise ResourceLimitError('controlled interruption')
            return self.producer(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'study'
            with patch('traxis.elmer_plate_refinement.run_elmer_waveform_case', side_effect=interrupted):
                with self.assertRaises(ResourceLimitError):
                    run_elmer_plate_refinement(local_scenario(intact=True), ElmerPlateRefinementSettings(), Path('unused'), output)
            saved = json.loads((output / 'refinement-report.json').read_text())
            self.assertFalse(saved['completed'])
            self.assertFalse(saved['discretization_screen_passed'])
            self.assertEqual(len(saved['cases']), 2)

    def test_weak_reference_traces_cannot_pass_discrepancy_screens(self):
        def zero(*args, **kwargs):
            values, measured = self.producer(*args, **kwargs)
            values[:] = 0.
            path = args[3] / 'solver-traces.npz'
            np.savez_compressed(path, time_s=np.arange(args[0].acquisition.samples) * args[0].acquisition.sample_dt_s,
                                displacement_m=values)
            measured['trace_sha256'] = sha256_file(path)
            write_json(args[3] / 'waveform-case.json', measured, overwrite=True)
            return values, measured
        with tempfile.TemporaryDirectory() as directory:
            with patch('traxis.elmer_plate_refinement.run_elmer_waveform_case', side_effect=zero):
                result = run_elmer_plate_refinement(local_scenario(intact=True), ElmerPlateRefinementSettings(), Path('unused'), Path(directory) / 'study')
            self.assertTrue(result['completed'])
            self.assertFalse(result['spatial_screen_passed'])
            self.assertFalse(result['temporal_screen_passed'])
            self.assertEqual(result['spatial_comparisons'][-1]['defined_records'], 0)
