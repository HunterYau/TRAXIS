"""Manufactured traces exercise acceptance logic; no physical validation here."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from examples.embedded_wave import case_config
from examples.embedded_xyz_study import checked_launcher,checked_level,compare_pairs,compare_xyz,load_protocol
from examples.tensor_wave import PROJECT,read_json,sealed
from examples.verify_embedded_xyz_window import cpu_limits,verify_xyz
from traxis.storage import sha256_file,write_json


class XYZStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=load_protocol(PROJECT/'configs/embedded-xyz-proposal.json')
        p=cls.protocol;cls.t=np.arange(p['steps']+1)*(p['duration_s']/p['steps'])

    def levels(self,scatter_errors=(.02,.005,.001),incident_scale=1.):
        plain=[];damaged=[];t=self.t
        for i,(ei,es) in enumerate(zip((.02,.005,.001),scatter_errors)):
            incident=incident_scale*(np.sin(2*np.pi*60000*t)+ei*np.cos(2*np.pi*80000*t))[:,None]*[1.,.7,.2]
            scatter=(np.sin(2*np.pi*100000*t)+es*np.cos(2*np.pi*120000*t))[:,None]*[.5,.3,.2]
            for group,state,y in ((plain,False,incident),(damaged,True,incident+scatter)):
                group.append(dict(case=case_config('xyz_refinement',xyz_level=i,defective=state),
                    window=dict(time_s=t,displacement_m=y)))
        return plain,damaged

    def test_all_nine_receiver_checks_and_integer_sample_selection(self):
        a,b=self.levels();t,traces,rows=compare_pairs(a,b,self.protocol)
        self.assertTrue(all(r['passed'] for group in rows.values() for r in group))
        self.assertEqual(traces['scattering'].shape,(3,32769,3));self.assertEqual(t[-1],.0005)
        for i in range(3):
            expected=(b[i]['window']['displacement_m']-a[i]['window']['displacement_m'])[::15]
            np.testing.assert_array_equal(traces['scattering'][i],expected)

    def test_total_fields_cannot_hide_a_failed_defect_signal(self):
        a,b=self.levels(scatter_errors=(.4,.2,.1),incident_scale=1000.)
        rows=compare_pairs(a,b,self.protocol)[2]
        self.assertTrue(all(r['passed'] for r in rows['intact']+rows['recessed']))
        self.assertFalse(any(r['passed'] for r in rows['scattering']))

    def test_zero_signal_missing_rate_and_bad_receiver_cannot_pass(self):
        a,b=self.levels()
        for plain,damaged in zip(a,b):damaged['window']['displacement_m']=plain['window']['displacement_m']
        self.assertTrue(all(not r['passed'] and not r['rate_identifiable'] for r in compare_pairs(a,b,self.protocol)[2]['scattering']))
        a,_=self.levels();y=[level['window']['displacement_m'].copy() for level in a]
        y[1][:,2]+=.2*np.sin(2*np.pi*130000*self.t)
        rows=compare_xyz([self.t]*3,y,self.protocol)[2]
        self.assertTrue(rows[0]['passed'] and rows[1]['passed']);self.assertFalse(rows[2]['passed'])

    def test_complete_float64_grids_and_frozen_thresholds_are_required(self):
        a,_=self.levels();y=[v['window']['displacement_m'] for v in a]
        t=self.t.copy();t[7]+=1e-15
        with self.assertRaises(ValueError):compare_xyz([t,self.t,self.t],y,self.protocol)
        for change in ('shape','float32','complex','nan'):
            values=list(y);values[0]=values[0].copy()
            if change=='shape':values[0]=values[0][:-1]
            elif change=='float32':values[0]=values[0].astype(np.float32)
            elif change=='complex':values[0]=values[0].astype(complex)
            else:values[0][4,0]=np.nan
            with self.assertRaises(ValueError):compare_xyz([self.t]*3,values,self.protocol)
        protocol=deepcopy(self.protocol);protocol['finest_relative_change_limit']=.1
        with self.assertRaises(ValueError):compare_xyz([self.t]*3,y,protocol)

    def test_xy_only_and_old_time_controls_cannot_substitute_for_xyz(self):
        for c in (case_config('full',4,294912),case_config('h_refinement',h_level=2)):
            with self.assertRaises(ValueError):checked_level(c,2,False)
        c=case_config('xyz_refinement',xyz_level=1,defective=True)
        with self.assertRaises(ValueError):checked_level(c,1,False)
        with self.assertRaises(ValueError):checked_level(c,2,True)

    def launcher(self,root,cpu_passed=True):
        limits=asdict(cpu_limits());proof=root/'cpu-window-check.json'
        write_json(proof,sealed(dict(completed=True,passed=cpu_passed,ready_for_training_data=False,
            training_updates=0,resources=dict(limits=limits))),overwrite=True)
        receipt=dict(kind='embedded_xyz_cpu_launcher',completed=True,passed=True,ready_for_training_data=False,
            training_updates=0,cpu_proof_sha256=sha256_file(proof),
            launcher_sha256=sha256_file(PROJECT/'examples/verify_embedded_xyz_window.py'),limits=limits)
        write_json(root/'cpu-launcher.json',sealed(receipt),overwrite=True)
        return receipt

    def test_launcher_requires_passing_cpu_wrapper_source_hash_and_matching_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.launcher(root);checked_launcher(root)
            for change in (lambda r:r.update(passed=False),lambda r:r.update(launcher_sha256='changed'),
                           lambda r:r.update(cpu_proof_sha256='changed'),
                           lambda r:r['limits'].update(max_rss_mib=8192)):
                receipt=self.launcher(root);receipt.pop('fingerprint',None);change(receipt)
                write_json(root/'cpu-launcher.json',sealed(receipt),overwrite=True)
                with self.assertRaises(ValueError):checked_launcher(root)
            self.launcher(root,cpu_passed=False)
            with self.assertRaises(ValueError):checked_launcher(root)

    def test_failed_outer_check_cannot_publish_a_passing_cpu_sidecar(self):
        with tempfile.TemporaryDirectory(dir=PROJECT/'outputs') as directory:
            root=Path(directory);run=root/'run';run.mkdir();output=root/'cpu'
            write_json(run/'embedded-wave-plan.json',dict(case=case_config('xyz_refinement',xyz_level=0)))
            def failing_verify(_run,out,_guard):
                out.mkdir();write_json(out/'cpu-window-check.json',sealed(dict(completed=True,passed=True)))
                raise ValueError('later failure')
            with patch('examples.verify_embedded_xyz_window.verify',side_effect=failing_verify):
                with self.assertRaisesRegex(ValueError,'later failure'):verify_xyz(run,output)
            sidecar=read_json(output/'cpu-launcher.json')
            self.assertFalse(sidecar['completed']);self.assertFalse(sidecar['passed'])


if __name__=='__main__':
    unittest.main()
