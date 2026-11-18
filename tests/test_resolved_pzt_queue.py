"""Long-run gate and automatic measured-comparison reporting."""
import unittest
from unittest.mock import patch
import numpy as np
from examples.resolved_pzt_queue import compare,run_queue


class QueueTests(unittest.TestCase):
    def test_held_queue_cannot_spawn_or_claim_ownership(self):
        with patch('examples.resolved_pzt_queue.configuration',return_value={'launch_ready':False}), \
                patch('examples.resolved_pzt_queue.write_json') as write, \
                patch('examples.resolved_pzt_queue.run_monitored') as run:
            with self.assertRaisesRegex(RuntimeError,'Long queue held'):run_queue()
        write.assert_not_called();run.assert_not_called()

    def test_identical_recordings_report_zero_and_preserve_unsupported_mode(self):
        t=np.arange(1751)*2e-7
        for f in (50000,250000):
            fast=np.exp(-((t-(48e-6+2.5/f))/8e-6)**2)*np.cos(2*np.pi*f*t)
            slow=np.exp(-((t-(130e-6+2.5/f))/15e-6)**2)*np.cos(2*np.pi*f*t)
            channels=np.column_stack((fast+slow,fast-slow))
            answer=compare(t,channels,channels,f,5000000)
            self.assertEqual(answer['fast_arrival_error_percent'],0.)
            if f==250000:
                self.assertIsNone(answer['slow_arrival_error_percent'])
                self.assertEqual(answer['slow_status'],'above_reported_A0_detection_range')


if __name__=='__main__':unittest.main()
