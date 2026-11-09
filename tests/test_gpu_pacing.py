import unittest
from unittest.mock import Mock

import psutil

from examples.pace_gpu_worker import CoolingPacer


class CoolingPacerTests(unittest.TestCase):
    def test_hysteresis_and_cleanup_resume_the_owned_worker(self):
        process = Mock()
        pacer = CoolingPacer(process)
        self.assertIsNone(pacer.update(77.))
        self.assertEqual(pacer.update(79.),'suspended')
        self.assertIsNone(pacer.update(76.))
        self.assertEqual(pacer.update(72.),'resumed')
        self.assertEqual(process.suspend.call_count,1)
        self.assertEqual(process.resume.call_count,1)
        pacer.update(78.)
        pacer.close()
        self.assertFalse(pacer.paused)
        self.assertEqual(process.resume.call_count,2)

    def test_failed_suspend_never_creates_a_false_paused_state(self):
        process = Mock()
        process.suspend.side_effect = psutil.AccessDenied(1)
        pacer = CoolingPacer(process)
        with self.assertRaises(psutil.AccessDenied):
            pacer.update(79.)
        self.assertFalse(pacer.paused)
        pacer.close()
        process.resume.assert_not_called()

    def test_cleanup_tolerates_a_worker_that_already_exited(self):
        process = Mock()
        pacer = CoolingPacer(process)
        pacer.update(79.)
        process.resume.side_effect = psutil.NoSuchProcess(1)
        pacer.close()
        self.assertFalse(pacer.paused)
