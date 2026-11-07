from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_forward import local_scenario
from traxis.config import CircularRecess, Plate
from traxis.highorder_directional import HighOrderDirectionalSettings
from traxis.highorder_waveforms import HighOrderWaveSettings, reference_actions, reference_mesh, run_highorder_waveform_case, run_native_job, screen_highorder_reference
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import canonical_hash, write_json


class HighOrderWaveformPreflightTests(unittest.TestCase):
    def test_recesses_require_explicit_cad_and_directional_cells_reject_cad(self):
        scenario = local_scenario()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'absent'
            with self.assertRaisesRegex(ValueError, 'requires the CAD'):
                run_highorder_waveform_case(scenario, HighOrderWaveSettings(), output, ResourceGuard(), state='current')
            self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, 'structured intact'):
            HighOrderDirectionalSettings(reference=HighOrderWaveSettings(mesh_backend='cad'))
        for changes in ({'mesh_backend': 'unknown'}, {'mesh_size_m': 0}, {'max_volume_error': 2},
                        {'max_tetrahedra': 40001}, {'mesh_algorithm': True}, {'mesh_optimization': 'unknown'}):
            with self.assertRaises(ValueError):
                HighOrderWaveSettings(**changes)

    def test_geometry_error_stops_before_highorder_assembly(self):
        scenario = local_scenario()
        settings = HighOrderWaveSettings(mesh_backend='cad')
        from traxis.geometry import structured_mesh
        mesh = structured_mesh(scenario.plate, (2, 2, 1))
        with patch('traxis.highorder_waveforms.cad_mesh', return_value=(mesh, {'removed_volume_relative_error': .2})):
            with self.assertRaisesRegex(ValueError, 'geometric error'):
                reference_mesh(scenario, settings, 'current', ResourceGuard())
        with self.assertRaisesRegex(ValueError, 'choose intact or current'):
            reference_actions(scenario, settings, state='unknown')

    def test_complete_grid_counts_and_allocation_limits(self):
        for order, dofs in ((1, 54), (2, 225), (3, 588), (4, 1215)):
            settings = HighOrderWaveSettings(order=order, cells=(2, 2, 1))
            self.assertEqual(screen_highorder_reference(settings)['vector_dofs'], dofs)
        with self.assertRaisesRegex(MemoryError, 'screen'):
            screen_highorder_reference(HighOrderWaveSettings(cells=(1000, 1000, 10)))
        for changes in ({'order': 5}, {'cells': (2, 0, 1)}, {'substeps_per_output': 65}, {'max_dofs': 60001}, {'max_assembly_mib': 513}):
            with self.assertRaises(ValueError):
                HighOrderWaveSettings(**changes)

    def test_noise_steps_and_invalid_actions_fail_before_output_creation(self):
        scenario, settings = local_scenario(intact=True), HighOrderWaveSettings(cells=(2, 2, 1))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'absent'
            noisy = replace(scenario, nuisance=replace(scenario.nuisance, relative_noise_std=.01))
            for case, config, kwargs in ((noisy, settings, {}), (scenario, replace(settings, max_internal_steps=1), {}),
                                        (scenario, settings, {'transmitter': True}), (scenario, settings, {'frequency_hz': 1.})):
                with self.assertRaises(ValueError):
                    run_highorder_waveform_case(case, config, output, ResourceGuard(), **kwargs)
            self.assertFalse(output.exists())
            self.assertEqual(len(reference_actions(scenario, settings)), 7)

    def test_unbound_or_changed_jobs_fail_without_importing_the_native_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = {'kind': 'incomplete'}
            job['fingerprint'] = canonical_hash(job)
            write_json(root / 'job.json', job)
            with self.assertRaisesRegex(ValueError, 'bindings'):
                run_native_job(root / 'job.json', root / 'native', ResourceGuard())
            job['kind'] = 'changed'
            write_json(root / 'job.json', job, overwrite=True)
            with self.assertRaisesRegex(ValueError, 'changed'):
                run_native_job(root / 'job.json', root / 'native', ResourceGuard())
            self.assertFalse((root / 'native').exists())


@unittest.skipUnless(os.environ.get('TRAXIS_HIGHORDER_CHECKS') == '1' and importlib.util.find_spec('ngsolve'),
                     'enable optional isolated high-order pulse checks')
class NativeHighOrderWaveformTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('gmsh'), 'install Gmsh for the CAD reference')
    def test_recessed_geometry_changes_mechanical_traces_and_preserves_cache_state(self):
        scenario = local_scenario()
        scenario = replace(scenario, plate=Plate(.02, .02, .003),
            recesses=(CircularRecess(.01, .01, .004, .0015),),
            acquisition=replace(scenario.acquisition, inset_m=.004, patch_radius_m=.0015),
            nuisance=replace(scenario.nuisance, current_gain_drift=tuple(1.1+.01*i for i in range(8))))
        settings = HighOrderWaveSettings(order=2, mesh_backend='cad', mesh_size_m=.002, substeps_per_output=4)
        guard = ResourceGuard(ResourceLimits(max_seconds=90, max_rss_mib=1024))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intact, baseline = run_highorder_waveform_case(scenario, settings, root/'intact', guard, cache_dir=root/'groups')
            current, damaged = run_highorder_waveform_case(scenario, settings, root/'current', guard, cache_dir=root/'groups', state='current')
            recovered, hit = run_highorder_waveform_case(scenario, settings, root/'recovered', guard, cache_dir=root/'groups', state='current')
            self.assertNotEqual(baseline['raw_cache_key'], damaged['raw_cache_key'])
            self.assertTrue(hit['cache_hit'])
            np.testing.assert_array_equal(current, recovered)
            self.assertEqual(hit['forward_solves'], 0)
            self.assertGreater(np.linalg.norm(current-intact)/np.linalg.norm(intact), .001)
            self.assertLess(damaged['mesh_geometry']['removed_volume_relative_error'], settings.max_volume_error)
            self.assertLess(damaged['mesh_geometry']['cad_retained_volume_m3'], baseline['mesh_geometry']['cad_retained_volume_m3'])
            self.assertIsNone(damaged['cells'])
            self.assertEqual(damaged['state'], 'current')
            with np.load(Path(damaged['native_directory'])/'traces.npz', allow_pickle=False) as saved:
                expected = saved['displacement_m']*np.array([1.1*(1.1+.01*i) for i in range(1,8)])[:,None]
                np.testing.assert_allclose(current, expected, rtol=2e-14, atol=0.)
            self.assertLess(damaged['rigid_body_checks']['maximum_translation_mass_relative_error'], 1e-10)
            self.assertLess(damaged['rigid_body_checks']['maximum_rigid_relative_force'], 1e-10)
            plan = json.loads((root/'current/case-plan.json').read_text())
            self.assertIn('gmsh', plan['runtime'])
            self.assertFalse(damaged['ready_for_training_data'])

    def test_mechanical_group_cache_and_input_artifact_integrity(self):
        settings = HighOrderWaveSettings(cells=(2, 2, 1), substeps_per_output=2)
        scenario = local_scenario(intact=True)
        guard = ResourceGuard(ResourceLimits(max_seconds=60, max_rss_mib=1024))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, measured = run_highorder_waveform_case(scenario, settings, root / 'first', guard, cache_dir=root / 'groups')
            second, hit = run_highorder_waveform_case(scenario, settings, root / 'second', guard, cache_dir=root / 'groups')
            self.assertEqual(first.shape, (7, 64))
            self.assertGreater(float(np.linalg.norm(first)), 1e-18)
            self.assertEqual(measured['forward_solves'], 1)
            self.assertEqual(hit['forward_solves'], 0)
            self.assertTrue(hit['cache_hit'])
            np.testing.assert_array_equal(first, second)
            self.assertEqual(measured['geometry_load_fingerprint'], hit['geometry_load_fingerprint'])
            self.assertFalse(hit['ready_for_training_data'])
            self.assertLess(measured['maximum_relative_linear_residual'], 1e-9)
            native = Path(measured['native_directory'])
            path = native / 'loads-probes.npz'
            original = path.read_bytes()
            path.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                run_highorder_waveform_case(scenario, settings, root / 'tampered', guard, cache_dir=root / 'groups')
            path.write_bytes(original)
            job_path = native.parent / 'worker-job.json'
            job = json.loads(job_path.read_text())
            job['scenario']['geometry_seed'] += 1
            write_json(job_path, job, overwrite=True)
            with self.assertRaisesRegex(ValueError, 'job changed'):
                run_highorder_waveform_case(scenario, settings, root / 'changed-job', guard, cache_dir=root / 'groups')
