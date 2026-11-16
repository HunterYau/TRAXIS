"""New stock defaults and explicit historical geometry are separate identities."""
from copy import deepcopy
from dataclasses import asdict,replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.tensor_wave import PROJECT,read_json
from examples.tensor_wave_study import case_for,case_levels,checked_protocol
from tests.fixtures import analytic_bank
from traxis.bank import WaveformBank
from traxis.config import Plate
from traxis.importers import import_trace_bundle
from traxis.models import ModelConfig
from traxis.scenarios import SamplingSettings,Scenario,sample_scenarios
from traxis.storage import canonical_hash


class StockDefaultsTests(unittest.TestCase):
    def test_new_default_and_unspecified_scenario_plans_use_eighth_inch_stock(self):
        self.assertEqual(Plate().thickness_m,.003175)
        self.assertEqual(ModelConfig().max_depth_mm,Plate().thickness_m*1000)
        self.assertEqual(ModelConfig(max_depth_mm=3.).max_depth_mm,3.)
        scenarios=sample_scenarios(SamplingSettings(count=2,intact_fraction=1))
        self.assertTrue(all(s.plate.thickness_m==.003175 for s in scenarios))
        recipe=read_json(PROJECT/'configs/scenarios-study.json')
        self.assertEqual(recipe['plate']['thickness_m'],.003175)
        self.assertEqual(recipe['sampling']['position_jitter_max_m'],.0005)
        self.assertTrue(recipe['sampling']['remount_between_states'])

    def test_historical_tensor_protocol_and_all_five_cases_keep_their_identities(self):
        fixture=read_json(Path(__file__).parent/'fixtures/stock-legacy-tensor-identities.json')
        protocol=checked_protocol(read_json(PROJECT/'configs/tensor-wave-protocol.json'))
        self.assertEqual(protocol['plate']['thickness_m'],.003)
        self.assertEqual(canonical_hash(protocol),fixture['protocol_fingerprint'])
        actual=[dict(order=o,steps=n,case_fingerprint=canonical_hash(case_for(protocol,o,n))) for o,n in case_levels(protocol)]
        self.assertEqual(actual,fixture['cases'])

    def test_complete_recorded_geometry_preserves_old_and_custom_thicknesses(self):
        for thickness in (.003,.003175,.001):
            plate=Plate(thickness_m=thickness)
            self.assertEqual(Plate.from_dict(asdict(plate)),plate)
        for bad in ({},{'width_m':.5,'height_m':.5},{**asdict(Plate()),'other':1}):
            with self.assertRaisesRegex(ValueError,'explicit width, height and thickness'):
                Plate.from_dict(bad)
        old=sample_scenarios(SamplingSettings(count=1,intact_fraction=1),Plate(thickness_m=.003))[0]
        self.assertEqual(Scenario.from_dict(old.to_dict()),old)
        partial=deepcopy(old.to_dict());partial['plate'].pop('thickness_m')
        with self.assertRaises(ValueError):Scenario.from_dict(partial)

    def test_old_bank_roundtrips_without_reinterpreting_its_plate(self):
        original=replace(analytic_bank(),plate=Plate(thickness_m=.003))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.npz';original.save(path);loaded=WaveformBank.load(path)
            self.assertEqual(loaded.plate.thickness_m,.003)
            np.testing.assert_array_equal(loaded.baseline,original.baseline)
            np.testing.assert_array_equal(loaded.waveforms,original.waveforms)
            with np.load(path,allow_pickle=False) as source:
                arrays={key:source[key] for key in source.files}
            manifest=json.loads(str(arrays['manifest'].item()));manifest['plate'].pop('thickness_m')
            arrays['manifest']=np.array(json.dumps(manifest));np.savez_compressed(path,**arrays)
            with self.assertRaisesRegex(ValueError,'recorded plate geometry'):WaveformBank.load(path)

    def test_incomplete_external_geometry_rejects_before_trace_file_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json'
            path.write_bytes(json.dumps(dict(schema_version=1,plate=dict(width_m=.5,height_m=.5))).encode())
            with self.assertRaisesRegex(ValueError,'recorded plate geometry'):import_trace_bundle(path)


if __name__=='__main__':
    unittest.main()
