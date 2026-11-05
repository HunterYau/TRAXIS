from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_elmer_plate_refinement as plate_tests
from tests.test_forward import local_scenario
from traxis.elmer_directional import ElmerDirectionalSettings, run_elmer_directional_refinement
from traxis.elmer_waveforms import ElmerWaveSettings
from traxis.storage import canonical_hash, sha256_file, write_json
from traxis.waveform_checks import plot_waveform_sweep


class DirectionalRefinementTests(unittest.TestCase):
    def producer(self, scenario, settings, home, output, guard, **kwargs):
        _, measured = plate_tests.PlateRefinementTests.producer(self, scenario, settings, home, output, guard, **kwargs)
        nx, ny, nz = settings.cells
        (output / 'mesh/mesh.nodes').write_text(json.dumps(settings.cells))
        phase = 2 * np.pi * 100000 * np.arange(scenario.acquisition.samples) * scenario.acquisition.sample_dt_s
        trace = 1e-9 * ((1 + 100 / nx**2) * np.sin(phase) + .02 / nz**2 * np.cos(phase)
                        + .1 / settings.substeps_per_output**2 * np.sin(2 * phase))
        values = np.tile(trace, (7, 1))
        np.savez_compressed(output / 'solver-traces.npz', time_s=np.arange(scenario.acquisition.samples) * scenario.acquisition.sample_dt_s,
                            displacement_m=values)
        measured['trace_sha256'] = sha256_file(output / 'solver-traces.npz')
        measured['quadratic_nodes'] = (2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1)
        measured['tetrahedra'] = nx * ny * nz * 6
        measured['linear_mesh'] = {'maximum_edge_m': float(np.sqrt((.04/nx)**2 + (.04/ny)**2 + (.003/nz)**2)),
                                   'maximum_element_z_span_m': .003 / nz}
        write_json(output / 'waveform-case.json', measured, overwrite=True)
        return values, measured

    def test_three_independent_axes_raw_discrepancies_and_checked_figures(self):
        settings = ElmerDirectionalSettings(in_plane_cells=(2, 3, 4))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'study'
            with patch('traxis.elmer_plate_refinement.run_elmer_waveform_case', side_effect=self.producer) as solve:
                report = run_elmer_directional_refinement(local_scenario(intact=True), settings, Path('unused'), output)
            definitions = [(call.args[1].cells, call.args[1].substeps_per_output) for call in solve.call_args_list]
            self.assertEqual(definitions, [((2, 2, 4), 16), ((3, 3, 4), 16), ((4, 4, 4), 16),
                                           ((4, 4, 1), 16), ((4, 4, 2), 16), ((4, 4, 4), 4), ((4, 4, 4), 8)])
            self.assertTrue(report['completed'])
            self.assertFalse(report['in_plane_screen_passed'])
            self.assertTrue(report['thickness_screen_passed'])
            self.assertTrue(report['temporal_screen_passed'])
            self.assertFalse(report['ready_for_training_data'])
            self.assertAlmostEqual(report['observed_thickness_order'], 2., places=8)
            self.assertAlmostEqual(report['observed_temporal_order'], 2., places=8)
            if importlib.util.find_spec('matplotlib'):
                figures = plot_waveform_sweep(output, Path(directory) / 'figures', (0,), directional=True)
                self.assertEqual(len(figures['files']), 4)
                path = output / 'case-00/solver-traces.npz'
                path.write_bytes(path.read_bytes() + b'changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    plot_waveform_sweep(output, Path(directory) / 'bad', (0,), directional=True)

    def test_invalid_axes_and_oversized_or_noisy_cases_fail_before_external_work(self):
        for values in ({'in_plane_cells': (10, 10, 20)}, {'thickness_cells': (1, 2, 3)},
                       {'substeps_per_output': (4, 8, 17)}, {'reference': ElmerWaveSettings()}):
            with self.assertRaises(ValueError):
                ElmerDirectionalSettings(**values)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'absent'
            scenario = local_scenario(intact=True)
            for settings, message in ((ElmerDirectionalSettings(max_total_solves=6), 'solve budget'),
                (ElmerDirectionalSettings(in_plane_cells=(10, 100, 1000)), 'structured P2'),
                (ElmerDirectionalSettings(reference=ElmerWaveSettings(mesh_backend='structured', max_internal_steps=1)), 'internal-step')):
                with self.assertRaisesRegex((ValueError, MemoryError), message):
                    run_elmer_directional_refinement(scenario, settings, Path('absent'), output)
            noisy = replace(scenario, nuisance=replace(scenario.nuisance, relative_noise_std=.01))
            with self.assertRaisesRegex(ValueError, 'zero stochastic noise'):
                run_elmer_directional_refinement(noisy, ElmerDirectionalSettings(), Path('absent'), output)
            self.assertFalse(output.exists())

    def test_changed_temporal_mesh_preserves_seven_cases_without_passing(self):
        def changed(*args, **kwargs):
            result = self.producer(*args, **kwargs)
            if args[3].name == 'case-06':
                (args[3] / 'mesh/mesh.nodes').write_text('changed mesh')
            return result
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'study'
            with patch('traxis.elmer_plate_refinement.run_elmer_waveform_case', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'spatial mesh'):
                    run_elmer_directional_refinement(local_scenario(intact=True), ElmerDirectionalSettings(in_plane_cells=(2, 3, 4)),
                                                       Path('unused'), output)
            report = json.loads((output / 'directional-report.json').read_text())
            self.assertFalse(report['completed'])
            self.assertFalse(report['discretization_screen_passed'])
            self.assertEqual(len(report['cases']), 7)
            self.assertEqual(report['fingerprint'], canonical_hash({k: v for k, v in report.items() if k != 'fingerprint'}))

    def test_returned_arrays_must_match_saved_comparison_artifacts(self):
        def changed(*args, **kwargs):
            values, measured = self.producer(*args, **kwargs)
            return values + 1e-9, measured
        with tempfile.TemporaryDirectory() as directory:
            with patch('traxis.elmer_plate_refinement.run_elmer_waveform_case', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'saved reference traces/time'):
                    run_elmer_directional_refinement(local_scenario(intact=True), ElmerDirectionalSettings(in_plane_cells=(2, 3, 4)),
                                                       Path('unused'), Path(directory) / 'study')
