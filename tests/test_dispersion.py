import unittest

import numpy as np

from traxis.config import Material
from traxis.dispersion import _determinant, mode_velocity, wavenumber_rad_m


class DispersionTests(unittest.TestCase):
    def test_thin_plate_extensional_and_flexural_limits(self):
        m = Material()
        f, thickness = 500., 0.003
        s = mode_velocity(f, thickness, "S0", m)
        a = mode_velocity(f, thickness, "A0", m)
        extensional = np.sqrt(m.young_pa / (m.density_kg_m3 * (1 - m.poisson**2)))
        rigidity = m.young_pa * thickness**3 / (12 * (1 - m.poisson**2))
        flexural = np.sqrt(2 * np.pi * f) * (rigidity / (m.density_kg_m3 * thickness))**0.25
        self.assertAlmostEqual(s.phase_m_s / extensional, 1, delta=1e-5)
        self.assertAlmostEqual(s.group_m_s / extensional, 1, delta=1e-5)
        self.assertAlmostEqual(a.phase_m_s / flexural, 1, delta=0.003)
        self.assertAlmostEqual(a.group_m_s / (2 * flexural), 1, delta=0.005)

    def test_traction_residual_and_order_over_research_band(self):
        material = Material()
        cl, ct = material.bulk_speeds_m_s
        for thickness in (0.001, 0.0015, 0.00225, 0.003):
            for f in (25000., 50000., 100000., 150000., 200000.):
                velocities = []
                for mode in ("A0", "S0"):
                    v = mode_velocity(f, thickness, mode, material)
                    residual = _determinant(v.wavenumber_rad_m * thickness / 2,
                                            np.pi * f * thickness / ct, ct / cl, mode)
                    self.assertLess(abs(residual), 1e-9)
                    self.assertGreater(v.group_m_s, 0)
                    self.assertAlmostEqual(v.wavelength_m * v.wavenumber_rad_m, 2 * np.pi)
                    velocities.append(v)
                self.assertLess(velocities[0].phase_m_s, ct)
                self.assertGreater(velocities[1].phase_m_s, ct)
                self.assertLess(velocities[1].phase_m_s, cl)

    def test_frequency_thickness_scaling_and_stable_group_derivative(self):
        for mode in ("A0", "S0"):
            a = mode_velocity(100000, 0.003, mode)
            b = mode_velocity(200000, 0.0015, mode, derivative_fraction=0.0005)
            self.assertAlmostEqual(a.phase_m_s, b.phase_m_s, places=7)
            self.assertAlmostEqual(a.group_m_s / b.group_m_s, 1, delta=1e-5)
            self.assertAlmostEqual(b.wavenumber_rad_m / a.wavenumber_rad_m, 2)

    def test_out_of_band_and_unknown_modes_fail(self):
        for f, thickness, mode in ((0, 0.003, "A0"), (1e6, 0.003, "A0"),
                                   (1, 0.003, "A0"), (1e5, 0.003, "S1")):
            with self.assertRaises(ValueError):
                wavenumber_rad_m(f, thickness, mode)

    def test_group_velocity_stencil_at_supported_band_edges(self):
        m = Material()
        ct = m.bulk_speeds_m_s[1]
        for fd in (1e-4, 0.25):
            for mode in ("A0", "S0"):
                v = mode_velocity(fd * ct / 0.003, 0.003, mode, m)
                self.assertGreater(v.group_m_s, 0)
