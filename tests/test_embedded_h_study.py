"""Manufactured spatial-error fields and separate CPU-launcher metadata tests."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from examples.embedded_h_study import checked_launcher,checked_level,compare_h,compare_pairs,load_protocol
from examples.embedded_wave import case_config
from examples.tensor_wave import PROJECT,checked_receipt,read_json,sealed
from examples.verify_embedded_h_window import verify_h
from traxis.storage import sha256_file,write_json


class HStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=load_protocol(PROJECT/'configs/embedded-h-protocol.json')
        p=cls.protocol;cls.t=np.arange(p['steps']+1)*(p['duration_s']/p['steps'])

    def levels(self,incident_errors=(.02,.005,.001),scatter_errors=(.02,.005,.001),incident_scale=1.):
        plain=[];damaged=[];t=self.t
        for i,(ei,es) in enumerate(zip(incident_errors,scatter_errors)):
            incident=incident_scale*(np.sin(2*np.pi*60000*t)+ei*np.cos(2*np.pi*80000*t))[:,None]*[1.,.7,.2]
            scatter=(np.sin(2*np.pi*100000*t)+es*np.cos(2*np.pi*120000*t))[:,None]*[.5,.3,.2]
            for group,state,y in ((plain,False,incident),(damaged,True,incident+scatter)):
                c=case_config('full',4,294912,state) if i==0 else case_config('h_refinement',h_level=i,defective=state)
                group.append(dict(case=c,window=dict(time_s=t,displacement_m=y)))
        return plain,damaged

    def test_original_coarse_and_new_h_levels_match_the_same_fixed_physics(self):
        a,b=self.levels();t,traces,rows=compare_pairs(a,b,self.protocol)
        self.assertTrue(all(r['passed'] for group in rows.values() for r in group))
        self.assertEqual(traces['scattering'].shape,(3,32769,3))
        self.assertEqual(t[-1],.0005)
        for i in range(3):
            expected=(b[i]['window']['displacement_m']-a[i]['window']['displacement_m'])[::9]
            np.testing.assert_array_equal(traces['scattering'][i],expected)

    def test_both_total_fields_can_pass_while_the_defect_signal_fails(self):
        a,b=self.levels(scatter_errors=(.4,.2,.1),incident_scale=1000.)
        rows=compare_pairs(a,b,self.protocol)[2]
        self.assertTrue(all(r['passed'] for r in rows['intact']+rows['recessed']))
        self.assertFalse(any(r['passed'] for r in rows['scattering']))

    def test_zero_defect_signal_does_not_establish_a_rate(self):
        a,b=self.levels()
        for plain,damaged in zip(a,b):damaged['window']['displacement_m']=plain['window']['displacement_m']
        rows=compare_pairs(a,b,self.protocol)[2]['scattering']
        self.assertTrue(all(not r['passed'] and not r['rate_identifiable'] for r in rows))
        self.assertTrue(all(r['finest_relative_change'] is None for r in rows))

    def test_every_receiver_must_decrease_and_meet_the_limit(self):
        a,b=self.levels();y=[l['window']['displacement_m'].copy() for l in a]
        y[1][:,2]+=.2*np.sin(2*np.pi*130000*self.t)
        rows=compare_h([self.t]*3,y,self.protocol)[2]
        self.assertTrue(rows[0]['passed'] and rows[1]['passed']);self.assertFalse(rows[2]['passed'])

    def test_full_grids_float64_and_unchanged_thresholds_are_required(self):
        a,_=self.levels();y=[l['window']['displacement_m'] for l in a]
        # A changed sample outside the selected stride must also reject.
        t=self.t.copy();t[4]+=1e-15
        with self.assertRaises(ValueError):compare_h([t,self.t,self.t],y,self.protocol)
        for change in ('shape','float32','complex','nan'):
            values=list(y);values[0]=values[0].copy()
            if change=='shape':values[0]=values[0][:-1]
            elif change=='float32':values[0]=values[0].astype(np.float32)
            elif change=='complex':values[0]=values[0].astype(complex)
            else:values[0][3,0]=np.nan
            with self.assertRaises(ValueError):compare_h([self.t]*3,values,self.protocol)
        p=deepcopy(self.protocol);p['finest_relative_change_limit']=.1
        with self.assertRaises(ValueError):compare_h([self.t]*3,y,p)

    def test_material_state_time_step_and_h_order_cannot_be_substituted(self):
        for case,index,state in [(case_config('full',4,196608),0,False),
                (case_config('h_refinement',h_level=2),1,False),
                (case_config('h_refinement',h_level=1,defective=True),1,False)]:
            with self.assertRaises(ValueError):checked_level(case,index,state)


class HCPUWrapperTests(unittest.TestCase):
    """Synthetic receipts test launcher behavior, not numerical CPU agreement."""
    def fixture(self,project):
        root=project/'outputs/run';root.mkdir(parents=True)
        write_json(root/'embedded-wave-plan.json',dict(case=case_config('h_refinement',h_level=1)))
        return root,project/'outputs/cpu'

    def test_original_checker_receipt_is_preserved_and_launcher_binds_it(self):
        for passed in (True,False):
            with tempfile.TemporaryDirectory() as temp:
                project=Path(temp);root,out=self.fixture(project);observed={}
                def checker(r,o,guard):
                    observed['limits']=guard.report()['limits'];o.mkdir()
                    report=sealed(dict(completed=True,passed=passed,fixture=True))
                    write_json(o/'cpu-window-check.json',report)
                    observed['proof']=(o/'cpu-window-check.json').read_bytes();return report
                with patch('examples.verify_embedded_h_window.PROJECT',project), \
                        patch('examples.verify_embedded_h_window.verify',side_effect=checker):
                    report=verify_h(root,out)
                self.assertEqual(report['passed'],passed)
                self.assertEqual((out/'cpu-window-check.json').read_bytes(),observed['proof'])
                launch=checked_receipt(read_json(out/'cpu-launcher.json'))
                self.assertEqual(launch['passed'],passed)
                self.assertEqual(launch['cpu_proof_sha256'],sha256_file(out/'cpu-window-check.json'))
                self.assertEqual(observed['limits']['max_system_memory_percent'],89.9)
                self.assertEqual(observed['limits']['max_seconds'],600)
                self.assertEqual(observed['limits']['max_rss_mib'],4096)

    def test_h_assessment_requires_the_passing_launcher_and_its_exact_cpu_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            project=Path(temp);root,out=self.fixture(project)
            def checker(r,o,g):
                o.mkdir();report=sealed(dict(completed=True,passed=True,fixture=True,resources=g.report()))
                write_json(o/'cpu-window-check.json',report);return report
            with patch('examples.verify_embedded_h_window.PROJECT',project), \
                    patch('examples.verify_embedded_h_window.verify',side_effect=checker):verify_h(root,out)
            case=case_config('h_refinement',h_level=1)
            self.assertIsNotNone(checked_launcher(case,out))
            original=checked_receipt(read_json(out/'cpu-launcher.json'))
            for key,value in [('completed',False),('passed',False),('ready_for_training_data',True),
                              ('training_updates',1),('cpu_proof_sha256','other'),('launcher_sha256','other'),('limits',{})]:
                changed={k:v for k,v in original.items() if k!='fingerprint'};changed[key]=value
                write_json(out/'cpu-launcher.json',sealed(changed),overwrite=True)
                with self.assertRaises(ValueError):checked_launcher(case,out)
            cpu={k:v for k,v in read_json(out/'cpu-window-check.json').items() if k!='fingerprint'}
            cpu['resources']['limits']['max_system_memory_percent']=95
            write_json(out/'cpu-window-check.json',sealed(cpu),overwrite=True)
            launch={k:v for k,v in original.items() if k!='fingerprint'}
            launch['cpu_proof_sha256']=sha256_file(out/'cpu-window-check.json')
            write_json(out/'cpu-launcher.json',sealed(launch),overwrite=True)
            with self.assertRaises(ValueError):checked_launcher(case,out)
            self.assertIsNone(checked_launcher(case_config('full',4,294912),out))

    def test_checker_failure_is_not_reported_as_a_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            project=Path(temp);root,out=self.fixture(project)
            def checker(r,o,g):
                o.mkdir();write_json(o/'cpu-window-check.json',sealed(dict(completed=False,passed=False)))
                raise RuntimeError('fixture checker stopped')
            with patch('examples.verify_embedded_h_window.PROJECT',project), \
                    patch('examples.verify_embedded_h_window.verify',side_effect=checker):
                with self.assertRaisesRegex(RuntimeError,'fixture checker stopped'):verify_h(root,out)
            launch=checked_receipt(read_json(out/'cpu-launcher.json'))
            self.assertFalse(launch['completed'] or launch['passed'])

    def test_changed_launcher_cannot_publish_success_after_a_passing_cpu_check(self):
        with tempfile.TemporaryDirectory() as temp:
            project=Path(temp);root,out=self.fixture(project);calls=[]
            def checker(r,o,g):
                o.mkdir();report=sealed(dict(completed=True,passed=True,fixture=True))
                write_json(o/'cpu-window-check.json',report);return report
            def digest(path):
                if Path(path).name=='verify_embedded_h_window.py':
                    calls.append(path);return 'first' if len(calls)==1 else 'changed'
                return sha256_file(path)
            with patch('examples.verify_embedded_h_window.PROJECT',project), \
                    patch('examples.verify_embedded_h_window.verify',side_effect=checker), \
                    patch('examples.verify_embedded_h_window.sha256_file',side_effect=digest):
                with self.assertRaisesRegex(ValueError,'launcher changed'):verify_h(root,out)
            launch=checked_receipt(read_json(out/'cpu-launcher.json'))
            self.assertFalse(launch['completed'] or launch['passed'])
            self.assertTrue(checked_receipt(read_json(out/'cpu-window-check.json'))['passed'])


if __name__=='__main__':unittest.main()
