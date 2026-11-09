import copy
import json
from pathlib import Path
import unittest

import numpy as np

from examples.layered_spatial_study import checked_configuration,common_grid,spatial_differences


class LayeredStudyTests(unittest.TestCase):
    def test_common_grid_is_stable_exact_and_never_shortens_the_window(self):
        limits = [6.314875851807e-8,3.564656492410e-8,1.907396511358e-8]
        steps,dt,fractions = common_grid(.0005,limits)
        self.assertEqual(steps,65536)
        self.assertEqual(dt*steps,.0005)
        for limit,cfl in zip(limits,fractions):
            self.assertLessEqual(cfl,.4)
            self.assertEqual(int(np.ceil(.0005/(limit*cfl))),steps)
        with self.assertRaises(MemoryError):
            common_grid(.0005,[1e-12]*3)

    def test_each_receiver_must_pass_the_unchanged_five_percent_screen(self):
        steps = 64
        time = np.arange(steps+1)*(.0005/steps)
        signal = np.sin(2*np.pi*time/.0005)
        traces = [np.column_stack([signal*(1+error),1e-9*signal*(1+20*error)]) for error in (.1,.025,0.)]
        rows = spatial_differences(time,traces,steps)
        self.assertTrue(rows[0]['passed'])
        self.assertFalse(rows[1]['passed'])
        self.assertAlmostEqual(rows[0]['finest_relative_change'],.025)
        self.assertAlmostEqual(rows[1]['finest_relative_change'],.5)
        with self.assertRaises(ValueError):
            spatial_differences(time,[a[:-1] for a in traces],steps)
        with self.assertRaises(ValueError):
            spatial_differences(time,[np.empty((steps+1,0))]*3,steps)
        with self.assertRaises(ValueError):
            spatial_differences(time.astype(complex),traces,steps)

    def test_study_criteria_and_distinct_levels_cannot_be_relaxed(self):
        config = json.loads((Path(__file__).resolve().parents[1]/'configs/layered-thickness-study.json').read_text())
        checked_configuration(config)
        changed = copy.deepcopy(config)
        changed['maximum_finest_relative_change'] = .5
        with self.assertRaises(ValueError):
            checked_configuration(changed)
        changed = copy.deepcopy(config)
        changed['levels'][1]['wave'] = changed['levels'][0]['wave']
        with self.assertRaises(ValueError):
            checked_configuration(changed)
