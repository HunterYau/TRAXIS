import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from examples.pzt_convergence import cases,metrics,energy_tail,prepare_case
from traxis.storage import sha256_file
from traxis.gpu_tensor import make_layout
from traxis.config import Plate


class ConvergenceTests(unittest.TestCase):
    def test_resume_preserves_interrupted_attempt_and_skips_verified_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'plate-200';folder.mkdir()
            (folder/'worker.log').write_text('interrupted at 2%')
            with patch('examples.pzt_convergence.worker_alive',return_value=False):
                self.assertTrue(prepare_case(folder))
                history=folder/'previous-attempts/attempt-001-unfinished'
                self.assertEqual((history/'worker.log').read_text(),'interrupted at 2%')
                self.assertTrue((history/'INTERRUPTED.txt').exists())
                waveform=folder/'waveforms.npz';waveform.write_bytes(b'saved test waveform')
                (folder/'result.json').write_text(json.dumps(dict(completed=True,
                    waveforms_sha256=sha256_file(waveform))))
                self.assertFalse(prepare_case(folder))
                self.assertTrue(waveform.exists())

    def test_resume_refuses_live_worker_and_uninvestigated_numerical_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with patch('examples.pzt_convergence.worker_alive',return_value=True):
                with self.assertRaises(RuntimeError):prepare_case(folder)
            (folder/'result.json').write_text(json.dumps(dict(completed=False,
                error='ArithmeticError: bad numerical result')))
            with patch('examples.pzt_convergence.worker_alive',return_value=False):
                with self.assertRaises(RuntimeError):prepare_case(folder)

    def test_cases_isolate_space_time_body_and_fit_layout_screens(self):
        rows={c['name']:c for c in cases()}
        for c in rows.values():
            make_layout(Plate(1,1,.001),(c['cells'],c['cells'],c.get('nz',1)),5,
                        max_dofs=32000000,max_device_mib=6144,max_host_mib=4096)
            self.assertEqual(c['steps']%(8*(c['steps']//65536)),0)
            s,r,l=c['body'];self.assertLessEqual((1+s*r)*(l+1),1024)
        for c in rows.values():
            if c['name'].startswith('plate-'):
                self.assertEqual(c['body'],[32,6,3]);self.assertEqual(c['steps'],131072)
            if c['name'].startswith('body-'):
                self.assertEqual(c['cells'],250);self.assertEqual(c['steps'],131072)

    def test_raw_changes_expose_gain_polarity_and_zero_reference(self):
        t=np.arange(100)*.1;b=np.column_stack((np.sin(t),np.cos(t)))
        self.assertAlmostEqual(metrics(2*b,b)['top']['raw_l2_change_percent'],100)
        self.assertAlmostEqual(metrics(-b,b)['top']['raw_l2_change_percent'],200)
        self.assertEqual(metrics(b,b)['top']['raw_l2_change_percent'],0)
        self.assertIsNone(metrics(b,np.zeros_like(b))['top']['raw_l2_change_percent'])

    def test_energy_check_excludes_work_during_last_interpolated_drive_ramp(self):
        # Last nonzero drive is at 1 s but the interpolation is nonzero until 2 s.
        t=np.array([1.2,1.5,1.8,2.2,2.5,2.8])
        energy=np.array([2.,3.,4.,5.,5.,5.])
        check=energy_tail(t,energy,np.array([0.,1.,2.,3.]),[0.,1.,0.,0.],.01)
        self.assertEqual(check['drive_support_end_s'],2.)
        self.assertEqual(check['post_drive_energy_samples'],3)
        self.assertEqual(check['energy_relative_drift'],0.)
        energy[-1]=6.
        self.assertGreater(energy_tail(t,energy,np.array([0.,1.,2.,3.]),
                                      [0.,1.,0.,0.],.01)['energy_relative_drift'],.1)
