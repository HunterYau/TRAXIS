import unittest
import numpy as np
from examples.pzt_sampling import antialiased_samples
from examples.pzt_drive_reconstruction import reconstructed_drive


class SamplingTests(unittest.TestCase):
    def test_downsampling_rejects_a_tone_that_would_alias_to_a0_band(self):
        t=np.arange(20000)/20000000;target=np.arange(5000)/5000000
        y=np.cos(2*np.pi*5250000*t)
        raw=np.interp(target,t,y)
        filtered=antialiased_samples(t,y,target)
        self.assertGreater(np.std(raw[100:-100]),.5)
        self.assertLess(np.std(filtered[100:-100]),1e-4)

    def test_in_band_voltage_amplitude_phase_and_same_rate_are_preserved(self):
        t=np.arange(20000)/20000000;target=np.arange(5000)/5000000
        y=2*np.cos(2*np.pi*250000*t+.37)
        expected=2*np.cos(2*np.pi*250000*target+.37)
        actual=antialiased_samples(t,y,target)
        np.testing.assert_allclose(actual[100:-100],expected[100:-100],atol=1e-5)
        np.testing.assert_array_equal(antialiased_samples(target,expected,target),expected)

    def test_drive_candidate_rejects_reconstruction_images_without_fitting_receiver(self):
        t=np.arange(5000)/5000000;v=np.sin(2*np.pi*250000*t)
        value,info=reconstructed_drive(t,v)
        grid=np.arange(10000)/20000000
        reconstructed=value(grid)
        # Interior samples reproduce a known analytic signal, avoiding FIR boundaries.
        np.testing.assert_allclose(reconstructed[200:-200],np.sin(2*np.pi*250000*grid[200:-200]),atol=4e-5)
        self.assertFalse(info['hardware_calibrated'])
        self.assertTrue(info['no_received_signal_fitting'])
