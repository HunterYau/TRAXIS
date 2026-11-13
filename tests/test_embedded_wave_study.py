"""Refinement rates and immutable raw grids on manufactured waveforms."""
from copy import deepcopy
import unittest

import numpy as np

from examples.embedded_wave_study import PROTOCOL, compare


class EmbeddedStudyTests(unittest.TestCase):
    def levels(self, mode, errors=None):
        counts=PROTOCOL['time_steps'] if mode=='time' else [PROTOCOL['spatial_steps']]*3
        times=[np.arange(n+1)*(PROTOCOL['duration_s']/n) for n in counts]
        if errors is None:
            errors=[1e13*(PROTOCOL['duration_s']/n)**2 for n in counts] if mode=='time' else [.02,.005,.001]
        traces=[(np.sin(2*np.pi*100000*t)+error*np.cos(2*np.pi*120000*t))[:,None]*[1.,.7,.2]
                for t,error in zip(times,errors)]
        return times,traces

    def test_second_order_at_ratio_one_point_five_on_integer_subsamples(self):
        t,y=self.levels('time'); common,aligned,rows=compare(t,y,'time',PROTOCOL)
        self.assertEqual(aligned.shape,(3,32769,3)); self.assertEqual(common[-1],.0005)
        self.assertTrue(all(row['passed'] for row in rows))
        for row in rows: self.assertAlmostEqual(row['observed_time_order'],2.,places=8)
        for i,n in enumerate(PROTOCOL['time_steps']):
            np.testing.assert_array_equal(aligned[i],y[i][::n//32768])

    def test_spatial_requires_identifiable_decreasing_small_raw_changes(self):
        t,y=self.levels('spatial'); _,_,rows=compare(t,y,'spatial',PROTOCOL)
        self.assertTrue(all(row['passed'] for row in rows))
        for errors in ([.002,.001,.02],[.1,.07,.05],[.001,.001,.001]):
            t,y=self.levels('spatial',errors); _,_,rows=compare(t,y,'spatial',PROTOCOL)
            self.assertFalse(any(row['passed'] for row in rows))

    def test_bad_time_rate_fails_even_with_small_changes(self):
        errors=[.0001,.00008,.00006]
        t,y=self.levels('time',errors); _,_,rows=compare(t,y,'time',PROTOCOL)
        self.assertFalse(any(row['passed'] for row in rows))

    def test_grid_protocol_shape_complex_and_nonfinite_changes_reject(self):
        t,y=self.levels('time'); altered=deepcopy(t); altered[1][20]+=1e-15
        with self.assertRaises(ValueError): compare(altered,y,'time',PROTOCOL)
        for change in ('shape','complex','nan'):
            changed=deepcopy(y)
            if change=='shape': changed[0]=changed[0][:-1]
            elif change=='complex': changed[0]=changed[0].astype(complex)
            else: changed[0][1,0]=np.nan
            with self.assertRaises(ValueError): compare(t,changed,'time',PROTOCOL)
        cfg=deepcopy(PROTOCOL);cfg['time_finest_change_limit']=1.
        with self.assertRaises(ValueError): compare(t,y,'time',cfg)


if __name__=='__main__': unittest.main()
