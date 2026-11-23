import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from examples.pzt_analytic_source import analytic_source
from examples.pzt_analytic_trial import run
from traxis.storage import sha256_file


class AnalyticTrialTests(unittest.TestCase):
    def test_exact_source_matches_samples_without_interpolating_or_fitting(self):
        t=np.arange(1000)/5000000;f=250000;duration=5/f
        drive=np.where(t<duration,np.sin(2*np.pi*f*t)*np.sin(np.pi*t/duration)**2,0.)
        force,info=analytic_source(f,5,t,drive)
        np.testing.assert_allclose(force(t),drive/max(abs(drive)),atol=1e-14)
        midpoint=.000008137
        expected=np.sin(2*np.pi*f*midpoint)*np.sin(np.pi*midpoint/duration)**2/max(abs(drive))
        self.assertAlmostEqual(force(midpoint),expected,places=14)
        self.assertEqual(force(-1e-6),0.);self.assertEqual(force(duration),0.)
        self.assertEqual(force(duration+1e-6),0.)
        self.assertFalse(info['received_signal_fitted'])
        with self.assertRaises(ValueError):analytic_source(f,5,t,drive*.9)

    def test_only_two_cases_run_in_order_and_save_aggregate_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);t=np.arange(8193)*.00035/8192
            v=np.column_stack((np.sin(2*np.pi*250000*t),.7*np.sin(2*np.pi*250000*t)))
            control=root/'control';control.mkdir()
            np.savez_compressed(control/'waveforms.npz',time_s=t,voltage_v=v)
            cases=[dict(name=n,steps=s,control_directory=str(control),
                        control_waveforms_sha256=sha256_file(control/'waveforms.npz'))
                   for n,s in [('analytic-original-dt',65536),('analytic-half-dt',131072)]]
            (root/'plan.json').write_text(json.dumps(dict(identity={'fixture':True},cases=cases)))
            order=[]
            def spawn(command,**kwargs):
                name=command[-1];order.append(name);folder=root/name
                if len(order)==2:
                    self.assertTrue((root/order[0]/'result.json').exists())
                    self.assertTrue((root/'findings.json').exists())
                np.savez_compressed(folder/'waveforms.npz',time_s=t,voltage_v=v)
                (folder/'result.json').write_text(json.dumps(dict(completed=True,
                    waveforms_sha256=sha256_file(folder/'waveforms.npz'))))
                return SimpleNamespace(pid=12345,stdout=['saved\n'],wait=lambda:0)
            with patch('examples.pzt_analytic_trial.identity',return_value={'fixture':True}), \
                    patch('examples.pzt_analytic_trial.subprocess.Popen',side_effect=spawn), \
                    patch('examples.pzt_analytic_trial.psutil.Process',return_value=SimpleNamespace(create_time=lambda:1.)), \
                    contextlib.redirect_stdout(io.StringIO()):
                run(root)
            self.assertEqual(order,[c['name'] for c in cases])
            self.assertEqual(json.loads((root/'status.json').read_text())['status'],'completed')
            self.assertTrue(json.loads((root/'findings.json').read_text())['completed'])
