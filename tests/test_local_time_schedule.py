"""Readiness and corruption checks, without constructing a full plate."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from examples import pzt_local_time_study as study
import json


class ScheduleTests(unittest.TestCase):
    def plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            with patch.object(study,'identity',return_value={}):
                study.prepare(root,2,[50000,100000],0.,[])
            return json.loads((root/'plan.json').read_text())

    def test_preparation_never_constructs_gpu_system(self):
        with patch.object(study.queue,'build',side_effect=AssertionError('GPU construction during preparation')):
            plan=self.plan()
        study.validate_plan(plan)
        self.assertFalse(plan['simulation_started'])
        self.assertEqual(len(plan['tasks']),7)
        self.assertEqual(plan['tasks'][0],'control-kernel')

    def test_changed_timestep_mesh_and_missing_control_are_rejected(self):
        plan=self.plan()
        for changed in ('time','mesh','tasks','config'):
            p=copy.deepcopy(plan)
            if changed=='time':p['coarse_dt_s']*=2
            elif changed=='mesh':p['case']['cells']-=1
            elif changed=='tasks':p['tasks'].remove('control-kernel')
            else:p['configuration']['mesh']['order']-=1
            with self.assertRaises(ValueError):study.validate_plan(p)
