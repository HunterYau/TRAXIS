import unittest
import numpy as np
from examples.diagnose_pzt_phase import phase_diagnostic


class PhaseDiagnosticTests(unittest.TestCase):
    def test_known_quadrature_shift_is_identified_without_changing_inputs(self):
        t=np.arange(1000)*2*np.pi/100
        simulated=np.cos(t)
        measured=3*np.cos(t+.7)
        before=simulated.copy()
        result=phase_diagnostic(measured,simulated)
        self.assertAlmostEqual(result['phase_rotation_degrees'],np.degrees(.7))
        self.assertAlmostEqual(result['fitted_gain'],3)
        self.assertLess(result['diagnostic_phase_fit']['normalized_wiggle_difference_percent'],1e-10)
        self.assertGreater(result['raw']['normalized_wiggle_difference_percent'],60)
        np.testing.assert_array_equal(before,simulated)

    def test_missing_signal_does_not_get_a_phase_estimate(self):
        result=phase_diagnostic(np.zeros(100),np.ones(100))
        self.assertIsNone(result['phase_rotation_degrees'])
        self.assertIsNone(result['diagnostic_phase_fit'])

    def test_different_waveform_remains_different_after_phase_fit(self):
        t=np.arange(1000)*2*np.pi/100
        result=phase_diagnostic(np.cos(t)+np.cos(2*t),np.cos(t))
        self.assertGreater(result['diagnostic_phase_fit']['normalized_wiggle_difference_percent'],70)
