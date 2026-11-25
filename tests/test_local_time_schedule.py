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
        self.assertEqual(len(plan['tasks']),2)
        self.assertEqual(plan['tasks'][0],'control-kernel')
        self.assertEqual(plan['tasks'],['control-kernel','lts-kernel'])
        self.assertEqual(plan['plate_condition'],'intact')
        self.assertEqual(plan['edge_width_m'],.004)
        self.assertAlmostEqual(plan['fine_dt_s']*1e9,5.340576171875)
        self.assertAlmostEqual(plan['coarse_dt_s']*1e9,10.68115234375)
        self.assertEqual(plan['coarse_steps'],32768)
        self.assertEqual(plan['source_sampling'],'coarse')

    def test_changed_timestep_mesh_and_missing_control_are_rejected(self):
        plan=self.plan()
        for changed in ('time','mesh','tasks','config'):
            p=copy.deepcopy(plan)
            if changed=='time':p['coarse_dt_s']*=2
            elif changed=='mesh':p['case']['cells']-=1
            elif changed=='tasks':p['tasks'].remove('control-kernel')
            else:p['configuration']['mesh']['order']-=1
            with self.assertRaises(ValueError):study.validate_plan(p)

    def test_old_refinement_schedule_is_rejected(self):
        plan=self.plan();plan['schema']=1
        plan['coarse_dt_s']=plan['fine_dt_s']
        with self.assertRaises(ValueError):study.validate_plan(plan)

    def test_all_six_frequencies_and_unmeasured_300khz_are_prepared(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(study,'identity',return_value={}):
            root=Path(tmp)/'six';study.prepare(root,2,study.FREQUENCIES,0.,[])
            plan=json.loads((root/'plan.json').read_text());study.validate_plan(plan)
        self.assertEqual(plan['frequencies_hz'],[50000,100000,150000,200000,250000,300000])
        with patch.object(study.baseline,'source',side_effect=AssertionError('No measured 300 kHz source')):
            source,info=study.excitation(300000,65536)
            self.assertEqual(source(0.),0.)
            self.assertEqual(source(5/300000),0.)
            self.assertFalse(info['experimental_ground_truth'])
