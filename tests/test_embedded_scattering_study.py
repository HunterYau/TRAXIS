"""Manufactured controls expose errors hidden by a large incident signal."""
from copy import deepcopy
import unittest

import numpy as np

from examples.embedded_scattering_study import paired_difference
from examples.embedded_wave import case_config
from examples.embedded_wave_study import PROTOCOL, compare


class ScatteringStudyTests(unittest.TestCase):
    def levels(self,mode,errors):
        counts=PROTOCOL['time_steps'] if mode=='time' else [PROTOCOL['spatial_steps']]*3
        orders=[4]*3 if mode=='time' else PROTOCOL['spatial_orders']
        intact=[];recessed=[]
        for n,p,e in zip(counts,orders,errors):
            t=np.arange(n+1)*(PROTOCOL['duration_s']/n)
            incident=np.sin(2*np.pi*60000*t)[:,None]*[1000.,800.,500.]
            scatter=(np.sin(2*np.pi*100000*t)+e*np.cos(2*np.pi*120000*t))[:,None]*[1.,.7,.2]
            for group,damaged,y in ((intact,False,incident),(recessed,True,incident+scatter)):
                group.append(dict(case=case_config('full',p,n,damaged),window=dict(time_s=t,displacement_m=y)))
        return intact,recessed

    def test_large_incident_wave_cannot_hide_nonconverged_scattering(self):
        intact,recessed=self.levels('spatial',[.4,.2,.1])
        t=[l['window']['time_s'] for l in recessed];y=[l['window']['displacement_m'] for l in recessed]
        self.assertTrue(all(r['passed'] for r in compare(t,y,'spatial',PROTOCOL)[2]))
        rows=paired_difference(intact,recessed,'spatial')[2]
        self.assertFalse(any(r['passed'] for r in rows))
        self.assertTrue(all(r['difference_ratio']>1 for r in rows))

    def test_second_order_time_difference_survives_paired_subtraction(self):
        errors=[1e13*(PROTOCOL['duration_s']/n)**2 for n in PROTOCOL['time_steps']]
        intact,recessed=self.levels('time',errors)
        t,y,rows=paired_difference(intact,recessed,'time')
        self.assertTrue(all(r['passed'] for r in rows))
        for r in rows:self.assertAlmostEqual(r['observed_time_order'],2.,places=5)
        self.assertEqual(y.shape,(3,32769,3));self.assertEqual(t[-1],.0005)

    def test_identical_states_do_not_claim_identifiable_scattering(self):
        intact,recessed=self.levels('spatial',[0.,0.,0.])
        for a,b in zip(intact,recessed):b['window']['displacement_m']=a['window']['displacement_m'].copy()
        rows=paired_difference(intact,recessed,'spatial')[2]
        self.assertFalse(any(r['passed'] or r['rate_identifiable'] for r in rows))
        self.assertTrue(all(r['finest_relative_change'] is None for r in rows))

    def test_changed_state_grid_and_cases_reject(self):
        intact,recessed=self.levels('spatial',[.02,.005,.001])
        self.assertTrue(all(r['passed'] for r in paired_difference(intact,recessed,'spatial')[2]))
        for change in ('grid','state','case','shape'):
            changed=deepcopy(recessed)
            if change=='grid':changed[0]['window']['time_s'][4]+=1e-13
            elif change=='state':changed[0]['case']['defective']=False
            elif change=='case':changed[0]['case']['radius_m']*=1.01
            else:changed[0]['window']['displacement_m']=changed[0]['window']['displacement_m'][:-1]
            with self.assertRaises(ValueError):paired_difference(intact,changed,'spatial')
        for group in (intact,recessed):group[0]['case']=case_config('full',4,196608,group is recessed)
        with self.assertRaises(ValueError):paired_difference(intact,recessed,'spatial')


if __name__=='__main__':unittest.main()
