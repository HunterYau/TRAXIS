"""Known packet timing, noise, competing packets and reflection rejection."""
import unittest
import numpy as np
from traxis.arrival import adaptive_packet
from examples.experimental_five import MATERIAL
from examples.compare_pzt_recordings import packet


class AdaptiveArrivalTests(unittest.TestCase):
    t=np.arange(1751)*2e-7

    def pulse(self,center,frequency,width=3e-6):
        return np.exp(-((self.t-center)/width)**2)*np.cos(2*np.pi*frequency*self.t)

    def test_recovers_packet_past_old_gate_without_hiding_timing_shift(self):
        f=200000;center=76e-6
        signal=self.pulse(center,f)
        fixed=packet(self.t,signal,f,'S0')
        adaptive=adaptive_packet(self.t,signal,f,'S0')
        self.assertTrue(fixed['window_edge'])
        self.assertEqual(adaptive['status'],'localized_packet')
        self.assertAlmostEqual(adaptive['packet_peak_us'],center*1e6,delta=.3)
        self.assertGreater(abs(adaptive['flight_us']-adaptive['predicted_flight_us']),10.)

    def test_late_stronger_reflection_does_not_replace_direct_packet(self):
        f=200000;signal=self.pulse(62e-6,f)+10*self.pulse(220e-6,f)
        answer=adaptive_packet(self.t,signal,f,'S0')
        self.assertEqual(answer['status'],'localized_packet')
        self.assertAlmostEqual(answer['packet_peak_us'],62.,delta=.3)

    def test_two_comparable_packets_are_unresolved(self):
        f=200000;signal=self.pulse(48e-6,f)+self.pulse(72e-6,f)
        answer=adaptive_packet(self.t,signal,f,'S0')
        self.assertEqual(answer['status'],'ambiguous_packets')

    def test_zero_and_noise_are_not_reliable_arrivals(self):
        for signal in (np.zeros(len(self.t)),np.random.default_rng(19).normal(size=len(self.t))):
            self.assertEqual(adaptive_packet(self.t,signal,200000,'S0')['status'],'no_signal')

    def test_windows_follow_each_recording_without_fitting_the_pair(self):
        first=adaptive_packet(self.t,self.pulse(60e-6,200000),200000,'S0')
        second=adaptive_packet(self.t,self.pulse(70e-6,200000),200000,'S0')
        self.assertAlmostEqual(second['packet_peak_us']-first['packet_peak_us'],10.,delta=.3)
        self.assertNotEqual(first['window_us'],second['window_us'])

    def test_comparison_does_not_fall_back_to_fixed_for_ambiguous_packets(self):
        from examples.resolved_pzt_queue import compare
        signal=self.pulse(48e-6,200000)+self.pulse(72e-6,200000)
        channels=np.column_stack((signal,signal))
        result=compare(self.t,channels,channels,200000,5000000)
        self.assertEqual(result['fast_arrival_method'],'adaptive')
        self.assertIsNone(result['fast_arrival_error_percent'])
        self.assertEqual(result['fast_status'],'unresolved_arrival')


if __name__=='__main__':unittest.main()
