import copy
import json
from pathlib import Path
import unittest

from examples.enriched_gpu_wave import checked_settings
from traxis.config import Plate


class EnrichedWindowPreflightTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((Path(__file__).resolve().parents[1]/'configs/enriched-gpu-wave.json').read_text())
        self.plate = Plate(.5,.5,.003)

    def test_full_observation_window_and_fixed_criteria_are_explicit(self):
        settings = checked_settings(self.settings,self.plate)
        self.assertEqual(settings['duration_s'],500e-6)
        self.assertEqual(settings['frequency_hz'],100000.)
        self.assertEqual(settings['cycles'],2)
        for key,value in (('cfl_fraction',.951),('operator_tolerance',1e-6),('energy_tolerance',1e-4),
                          ('duration_s',1e-6),('duration_s',2e-3)):
            with self.subTest(key=key,value=value):
                changed = copy.deepcopy(self.settings)
                changed[key] = value
                with self.assertRaises(ValueError):
                    checked_settings(changed,self.plate)

    def test_invalid_footprints_and_device_limits_are_rejected_without_native_work(self):
        for key,value in (('receiver_centers_xy_m',[]),('receiver_centers_xy_m',[[.02,.02]]),
                          ('receiver_centers_xy_m',[[.501,.2]]),('source_center_xy_m',[0.,.02]),
                          ('gpu',{'max_device_mib':1024,'chunk_elements':0,'device_id':0})):
            with self.subTest(key=key,value=value):
                changed = copy.deepcopy(self.settings)
                changed[key] = value
                with self.assertRaises(ValueError):
                    checked_settings(changed,self.plate)
        changed = copy.deepcopy(self.settings)
        changed['ignored_setting'] = True
        with self.assertRaises(ValueError):
            checked_settings(changed,self.plate)
