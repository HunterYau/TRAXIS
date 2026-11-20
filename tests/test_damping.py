import json
from pathlib import Path
import unittest

from traxis.damping import damping_parameters


class DampingTests(unittest.TestCase):
    def test_reference_conversion_and_frequency_scaling(self):
        profile=json.loads(Path('configs/pzt/aluminum-damping-ono2022.json').read_text())
        a=damping_parameters(profile,50000);b=damping_parameters(profile,250000)
        self.assertAlmostEqual(a['loss_factor'],.001172,places=5)
        self.assertAlmostEqual(b['mass_damping_per_s'],5*a['mass_damping_per_s'])
        self.assertFalse(a['specimen_calibrated'])
        for value in (-1.,0.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):damping_parameters(profile,value)
