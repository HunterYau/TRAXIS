from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_forward import local_scenario
from traxis.config import Plate
from traxis.forward import ForwardSettings, generate_waveform_bank
from traxis.mounting import PZTMounting
from traxis.scenarios import (AcquisitionSettings, SamplingSettings, Scenario, load_scenario_plan,
                              nuisance_variant, sample_scenarios, save_scenario_plan)
from traxis.solver_requests import export_solver_request
from traxis.storage import canonical_hash


class MountingRandomizationTests(unittest.TestCase):
    def settings(self, **overrides):
        return replace(SamplingSettings(count=20, position_jitter_max_m=.0005,
            contact_radius_relative_range=.1, remount_between_states=True), **overrides)

    def test_seeded_bounded_mounts_for_both_layouts_and_states(self):
        settings = self.settings()
        for count in (8, 12):
            acquisition = AcquisitionSettings(sensor_count=count)
            first = sample_scenarios(settings, Plate(thickness_m=.003175), acquisition)
            self.assertEqual(first, sample_scenarios(settings, Plate(thickness_m=.003175), acquisition))
            for scenario in first:
                self.assertNotEqual(scenario.mounting, scenario.current_mounting)
                for state, mount in (('intact', scenario.mounting), ('current', scenario.current_mounting)):
                    offsets = np.asarray(mount.offsets_xy_m)
                    self.assertLessEqual(np.linalg.norm(offsets, axis=1).max(), .0005*(1+1e-12))
                    self.assertGreater(np.linalg.norm(offsets), 0)
                    scales = np.asarray(mount.contact_radius_scales)
                    self.assertTrue(np.all((scales >= .9) & (scales <= 1.1)))
                    xy, radii = scenario.sensor_geometry(state)
                    self.assertTrue(np.all(xy >= radii[:, None]))
                    self.assertTrue(np.all(xy <= [.5, .5]-radii[:, None]))
                    for recess in scenario.recesses:
                        distance = np.linalg.norm(xy-[recess.x_m, recess.y_m], axis=1)
                        self.assertTrue(np.all(distance > recess.radius_m+radii+settings.sensor_clearance_m))
                current_gains = np.asarray(scenario.nuisance.sensor_gains)*scenario.nuisance.current_gain_drift
                self.assertTrue(np.all(current_gains >= .9*.98))
                self.assertTrue(np.all(current_gains <= 1.1*1.02))

    def test_variants_preserve_plate_material_parent_and_split(self):
        settings = self.settings()
        base = sample_scenarios(settings)[4]
        variant = nuisance_variant(base, settings, 713)
        self.assertEqual(variant, nuisance_variant(base, settings, 713))
        for field in ('plate', 'recesses', 'parent_specimen_id', 'split'):
            self.assertEqual(getattr(base, field), getattr(variant, field))
        self.assertEqual(base.nuisance.material, variant.nuisance.material)
        self.assertNotEqual(base.mounting, variant.mounting)
        self.assertNotEqual(base.specimen_id, variant.specimen_id)
        changed_gain = nuisance_variant(base, replace(settings, coupling_relative_range=.2), 713)
        self.assertEqual(variant.mounting, changed_gain.mounting)
        self.assertEqual(variant.current_mounting, changed_gain.current_mounting)
        self.assertNotEqual(variant.nuisance.sensor_gains, changed_gain.nuisance.sensor_gains)

    def test_roundtrip_and_legacy_nominal_fingerprint(self):
        nominal = local_scenario(intact=True)
        old = asdict(nominal); old.pop('mounting'); old.pop('current_mounting')
        self.assertEqual(nominal.fingerprint, canonical_hash(old))
        self.assertEqual(nominal.to_dict(), old)
        self.assertEqual(Scenario.from_dict(old), nominal)
        moved = replace(nominal, mounting=PZTMounting(((.0001, 0.),)+( (0., 0.),)*7, (1.,)*8))
        self.assertNotEqual(moved.fingerprint, nominal.fingerprint)
        randomized = nuisance_variant(nominal, self.settings(), 51)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mounts.json'
            report = save_scenario_plan(path, (nominal, randomized))
            self.assertEqual(report['schema_version'], 2)
            self.assertEqual(load_scenario_plan(path), (nominal, randomized))
            self.assertFalse(report['ready_for_training_data'])

    def test_invalid_mounts_and_ranges_are_rejected(self):
        for kwargs in (dict(position_jitter_max_m=-.001), dict(position_jitter_max_m=float('nan')),
                       dict(contact_radius_relative_range=1), dict(remount_between_states=1)):
            with self.assertRaises(ValueError):
                self.settings(**kwargs)
        for offsets, scales in [(((0., 0.),)*7, (1.,)*7), (((float('nan'), 0.),)*8, (1.,)*8),
                                 (((0., 0.),)*8, (0.,)*8)]:
            with self.assertRaises(ValueError):
                PZTMounting(offsets, scales)
        nominal = local_scenario(intact=True)
        for mount in (PZTMounting(((-.02, 0.),)*8, (1.,)*8),
                      PZTMounting(((0., 0.),)*8, (10.,)*8)):
            with self.assertRaises(ValueError):
                replace(nominal, mounting=mount)
        with self.assertRaisesRegex(ValueError, 'plate edge'):
            sample_scenarios(self.settings(count=1, intact_fraction=1, contact_radius_relative_range=0),
                Plate(.02, .02, .003175), AcquisitionSettings(inset_m=.004, patch_radius_m=.006))
        with self.assertRaisesRegex(ValueError, 'mounting'):
            sample_scenarios(self.settings(count=1, intact_fraction=1, position_jitter_max_m=1.),
                Plate(.02, .02, .003175), AcquisitionSettings(inset_m=.004, patch_radius_m=.002))

    def test_export_records_actual_ports_but_import_context_stays_nominal(self):
        scenario = nuisance_variant(local_scenario(intact=True), self.settings(), 31)
        with tempfile.TemporaryDirectory() as directory:
            task = export_solver_request(scenario, Path(directory))
            self.assertEqual(task['schema_version'], 2)
            spec = json.loads((Path(directory)/'import-manifest.json').read_text())
            np.testing.assert_array_equal(spec['sensors_xy_m'], scenario.acquisition.sensors(scenario.plate))
            for job in task['jobs']:
                xy, radii = scenario.sensor_geometry(job['state'])
                np.testing.assert_array_equal(job['actual_sensors_xy_m'], xy)
                np.testing.assert_array_equal(job['contact_radii_m'], radii)
            self.assertNotEqual(task['jobs'][0]['actual_sensors_xy_m'], task['jobs'][7]['actual_sensors_xy_m'])

    def test_ports_change_actual_waveforms_and_cache_identity(self):
        scenario = local_scenario(intact=True)
        scenario = replace(scenario, plate=replace(scenario.plate, thickness_m=.003175))
        settings = ForwardSettings(mesh_backend='structured')
        offsets = np.zeros((8, 2)); offsets[0] = [.0003, -.0002]
        moved = PZTMounting(tuple(map(tuple, offsets)), (1.,)*8)
        radius = PZTMounting(((0., 0.),)*8, (1.1,)+(1.,)*7)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)/'cache'
            base, _ = generate_waveform_bank(scenario, settings, cache_dir=cache)
            gain_scenario = replace(scenario, nuisance=replace(scenario.nuisance, sensor_gains=(1.1,)+(1.,)*7))
            with patch('traxis.forward.simulate', side_effect=AssertionError('gain-only change reran physics')):
                gained, hit = generate_waveform_bank(gain_scenario, settings, cache_dir=cache)
            self.assertEqual(hit['cache_hits'], 7)
            factors = np.array([1.1 if a.transmitter == 0 or a.receiver == 0 else 1. for a in base.actions])
            np.testing.assert_allclose(gained.waveforms, base.waveforms*factors[:, None], rtol=2e-7, atol=1e-20)
            for mount in (moved, radius):
                changed, report = generate_waveform_bank(replace(scenario, mounting=mount), settings, cache_dir=cache)
                self.assertEqual(report['forward_solves'], 7)
                self.assertGreater(np.linalg.norm(changed.waveforms-base.waveforms)/np.linalg.norm(base.waveforms), 1e-6)
                np.testing.assert_array_equal(changed.sensors_xy_m, scenario.acquisition.sensors(scenario.plate))
                self.assertFalse(report['ready_for_training_data'])
            remounted, report = generate_waveform_bank(replace(scenario, current_mounting=moved), settings, cache_dir=cache)
            self.assertEqual(report['cache_hits'], 7)
            self.assertEqual(report['forward_solves'], 7)
            self.assertGreater(np.linalg.norm(remounted.waveforms-remounted.baseline)/np.linalg.norm(remounted.baseline), 1e-6)
            self.assertFalse(remounted.depth_mm.any())


@unittest.skipUnless(os.environ.get('TRAXIS_HIGHORDER_CHECKS') == '1', 'enable the bounded native mounting check')
class NativeMountingTests(unittest.TestCase):
    def test_native_pulse_uses_actual_current_footprints(self):
        from traxis.highorder_waveforms import HighOrderWaveSettings, run_highorder_waveform_case
        from traxis.resources import ResourceGuard, ResourceLimits
        scenario = local_scenario(intact=True)
        offsets = np.zeros((8, 2)); offsets[0] = [.0004, -.0003]
        mount = PZTMounting(tuple(map(tuple, offsets)), (1.1,)+(1.,)*7)
        scenario = replace(scenario, plate=replace(scenario.plate, thickness_m=.003175), current_mounting=mount)
        settings = HighOrderWaveSettings(order=2, cells=(2, 2, 1))
        guard = ResourceGuard(ResourceLimits(max_seconds=60, max_rss_mib=768,
            max_system_memory_percent=85, min_available_mib=2048))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intact, first = run_highorder_waveform_case(scenario, settings, root/'intact', guard, cache_dir=root/'cache')
            current, second = run_highorder_waveform_case(scenario, settings, root/'current', guard,
                state='current', cache_dir=root/'cache')
            self.assertNotEqual(first['raw_cache_key'], second['raw_cache_key'])
            self.assertGreater(np.linalg.norm(current-intact)/np.linalg.norm(intact), 1e-6)
            self.assertFalse(second['ready_for_training_data'])


if __name__ == '__main__':
    unittest.main()
