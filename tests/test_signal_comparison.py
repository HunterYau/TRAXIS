import unittest
import numpy as np
from traxis.signal_comparison import waveform_metrics
from examples.resolved_pzt_queue import compare
from examples.pzt_terminal import result_lines


class SignalComparisonTests(unittest.TestCase):
    def test_gain_difference_is_visible_in_raw_metric(self):
        y=np.sin(np.arange(100)*.2)
        m=waveform_metrics(y,4*y)
        self.assertAlmostEqual(m['raw_voltage_difference_percent'],300.)
        self.assertLess(m['normalized_wiggle_difference_percent'],1e-12)
        self.assertAlmostEqual(m['correlation'],1.)

    def test_opposite_polarity_is_not_hidden_as_matching_wiggles(self):
        y=np.sin(np.arange(100)*.2);m=waveform_metrics(y,-y)
        self.assertAlmostEqual(m['normalized_wiggle_difference_percent'],200.)
        self.assertAlmostEqual(m['correlation'],-1.)

    def test_zero_signal_and_shift_are_not_reported_as_perfect_matches(self):
        self.assertIsNone(waveform_metrics(np.zeros(100),np.zeros(100))['correlation'])
        y=np.sin(np.arange(100)*.2)
        self.assertGreater(waveform_metrics(y,np.roll(y,4))['normalized_wiggle_difference_percent'],20.)
        for a,b in [(y,y[:-1]),(y,np.full(100,np.nan))]:
            with self.assertRaises(ValueError):waveform_metrics(a,b)

    def test_envelope_can_match_while_wiggles_disagree_and_results_are_printed(self):
        t=np.arange(1751)*2e-7;f=100000
        s=np.exp(-((t-80e-6)/15e-6)**2)*np.cos(2*np.pi*f*t)
        a=np.exp(-((t-160e-6)/20e-6)**2)*np.cos(2*np.pi*f*t)
        real=np.column_stack((s+a,s-a))
        c=compare(t,-real,real,f,5000000)
        for label in ('fast','slow'):
            self.assertLess(c[label+'_envelope_shape_error_percent'],1e-10)
            self.assertAlmostEqual(c[label+'_envelope_pearson_r'],1.)
            self.assertAlmostEqual(c[label+'_waveform']['normalized_wiggle_difference_percent'],200.)
        text='\n'.join(result_lines(f,{'completed':True,'comparison':c},'.'))
        self.assertIn('A0-related trace individual-wiggle difference: 200.00%',text)
        self.assertIn('raw voltage difference',text)
        self.assertIn('S0-related trace envelope Pearson r: 1.0000',text)
        self.assertIn('A0-related trace envelope Pearson r: 1.0000',text)

    def test_zero_envelopes_have_undefined_pearson_and_print_that_status(self):
        t=np.arange(1751)*2e-7;channels=np.zeros((len(t),2))
        c=compare(t,channels,channels,100000,5000000)
        self.assertIsNone(c['fast_envelope_pearson_r'])
        self.assertIsNone(c['slow_envelope_pearson_r'])
        text='\n'.join(result_lines(100000,{'completed':True,'comparison':c},'.'))
        self.assertIn('envelope Pearson r: undefined (constant or zero envelope)',text)
