import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.enriched_time_refinement import checked_archive,checked_plan,compare_traces,same_problem
from traxis.storage import sha256_file


def criteria():
    values = json.loads((Path(__file__).resolve().parents[1]/'configs/enriched-time-refinement.json').read_text())
    for row in values['levels']:
        row['steps'] = 16*row['factor']
    return values


def manufactured(order=(2,2),scales=(1.,.2)):
    plan = criteria()
    times,traces = [],[]
    for row in plan['levels']:
        time = np.linspace(0.,plan['duration_s'],row['steps']+1)
        phase = time/plan['duration_s']
        values = np.column_stack([scale*(np.sin(2*np.pi*phase)+(.1/row['steps'])**power*np.cos(6*np.pi*phase))
            for power,scale in zip(order,scales)])
        times.append(time)
        traces.append(values)
    return plan,times,traces


class FullWindowTimeComparisonTests(unittest.TestCase):
    def test_second_order_raw_records_match_without_interpolation(self):
        plan,times,traces = manufactured()
        rows,aligned = compare_traces(times,traces,plan)
        self.assertTrue(all(r['passed'] for r in rows))
        self.assertTrue(all(abs(r['observed_order']-2)<1e-8 for r in rows))
        for i in range(3):
            np.testing.assert_equal(aligned[i],traces[i][::2**i])
        np.testing.assert_allclose([r['finest_waveform_l2_m'] for r in rows],np.linalg.norm(aligned[-1],axis=0))
        changed = copy.deepcopy(plan)
        changed['ratio_bounds'] = [1.,10.]
        with self.assertRaises(ValueError):
            checked_plan(changed)

    def test_weak_receiver_cannot_hide_bad_temporal_order(self):
        plan,times,traces = manufactured(order=(2,1),scales=(1000.,1e-9))
        rows,_ = compare_traces(times,traces,plan)
        self.assertTrue(rows[0]['passed'])
        self.assertFalse(rows[1]['passed'])
        self.assertAlmostEqual(rows[1]['observed_order'],1.,places=6)

    def test_missing_or_shifted_samples_and_unidentifiable_rates_do_not_pass(self):
        plan,times,traces = manufactured()
        shifted = [t.copy() for t in times]
        shifted[1][3] += 1e-10
        with self.assertRaises(ValueError):
            compare_traces(shifted,traces,plan)
        with self.assertRaises(ValueError):
            compare_traces([times[0],times[1][:-1],times[2]],traces,plan)
        zero = [np.zeros_like(a) for a in traces]
        rows,_ = compare_traces(times,zero,plan)
        self.assertFalse(any(r['passed'] for r in rows))
        self.assertTrue(all(r['difference_ratio'] is None for r in rows))

    def test_changed_physics_or_code_are_rejected(self):
        base = dict(kind='window',mode='wave',input_sha256={'mesh':'same'},source_sha256={'fem':'same'},
            driver_sha256={'driver':'same'},runtime={'version':'same'},gpu_packages={'version':'same'},
            settings={'duration_s':.0005,'frequency_hz':100000.,'cfl_fraction':.8})
        half = copy.deepcopy(base)
        half['settings']['cfl_fraction'] = .4
        same_problem([base,half])
        half['settings']['frequency_hz'] = 200000.
        with self.assertRaises(ValueError):
            same_problem([base,half])
        half = copy.deepcopy(base)
        half['source_sha256']['fem'] = 'changed'
        with self.assertRaises(ValueError):
            same_problem([base,half])

    def test_changed_archive_is_rejected_before_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'comparison.npz'
            np.savez_compressed(path,signal=np.arange(10.))
            digest = sha256_file(path)
            checked_archive(path,digest)
            np.savez_compressed(path,signal=np.arange(11.))
            with self.assertRaisesRegex(ValueError,'changed'):
                checked_archive(path,digest)
