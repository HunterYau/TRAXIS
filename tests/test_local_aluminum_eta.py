"""ETA estimates use current-attempt progress, independent of simulation state."""
import unittest
from examples.local_aluminum_study import SimulationETA

class SimulationETATests(unittest.TestCase):
    def test_excludes_initial_capture_overhead(self):
        eta=SimulationETA(1000)
        self.assertIsNone(eta.update(0,None)['eta_remaining_seconds'])
        eta.update(16,100.);eta.update(32,102.)
        answer=eta.update(48,104.)
        self.assertEqual(answer['steps_per_second'],8.)
        self.assertEqual(answer['eta_remaining_seconds'],119.)
        self.assertEqual(answer['eta_remaining'],'00:01:59')
    def test_new_attempt_resets_rate(self):
        eta=SimulationETA(1000)
        for n,t in [(16,2.),(32,4.),(48,6.)]:eta.update(n,t)
        self.assertIsNone(eta.update(16,1.)['eta_remaining_seconds'])
        eta.update(32,2.);answer=eta.update(48,3.)
        self.assertEqual(answer['steps_per_second'],16.)
    def test_window_tracks_new_speed(self):
        eta=SimulationETA(10000)
        for i in range(1,81):answer=eta.update(16*i,float(i))
        for i in range(81,161):answer=eta.update(16*i,80.+2*(i-80))
        self.assertEqual(answer['steps_per_second'],8.)
        self.assertAlmostEqual(answer['eta_remaining_seconds'],(10000-2560)/8)
    def test_complete_and_degenerate_progress(self):
        eta=SimulationETA(100)
        for n in [1,2,3]:self.assertIsNone(eta.update(n,1.)['eta_remaining_seconds'])
        self.assertEqual(eta.update(100,None)['eta_remaining_seconds'],0.)
        self.assertIsNone(eta.update(0,None)['eta_remaining_seconds'])

if __name__=='__main__':unittest.main()
