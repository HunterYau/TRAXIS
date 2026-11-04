import unittest

from traxis.dispersion import wavenumber_rad_m
from traxis.waveguide import lamb_frequencies_hz


class WaveguideTests(unittest.TestCase):
    def test_fem_guided_modes_converge_to_rayleigh_lamb(self):
        for thickness in (0.001, 0.003):
            for mode, index in (("A0", 0), ("S0", 1)):
                target_f = 100000.
                k = wavenumber_rad_m(target_f, thickness, mode)
                errors = []
                for n in (16, 32, 64, 128):
                    f = lamb_frequencies_hz(k, thickness, n, n // 2)[index]
                    errors.append(abs(f / target_f - 1))
                for coarse, fine in zip(errors, errors[1:]):
                    self.assertGreater(coarse / fine, 3)
                self.assertLess(errors[-1], 0.005)

    def test_x_and_z_refinement_are_independently_stable(self):
        thickness, target = 0.003, 150000.
        k = wavenumber_rad_m(target, thickness, "A0")
        # Isolate spatial components rather than conflating x/z refinement.
        x_errors = [abs(lamb_frequencies_hz(k, thickness, n, 64)[0] / target - 1) for n in (8, 16, 32)]
        z_errors = [abs(lamb_frequencies_hz(k, thickness, 128, n)[0] / target - 1) for n in (4, 8, 16)]
        self.assertTrue(all(a > b for a, b in zip(x_errors, x_errors[1:])))
        self.assertTrue(all(a > b for a, b in zip(z_errors, z_errors[1:])))

    def test_bad_bloch_settings_fail(self):
        with self.assertRaises(ValueError):
            lamb_frequencies_hz(0, 0.003)
        with self.assertRaises(ValueError):
            lamb_frequencies_hz(100, 0.003, thickness_cells=1)
