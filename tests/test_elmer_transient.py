import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from traxis.config import Material
from traxis.elmer_transient import read_scalar_probe, sinusoidal_displacement, transient_sif, verify_elmer_transient
from traxis.resources import ResourceGuard, ResourceLimits


class AnalyticTransientTests(unittest.TestCase):
    def test_closed_form_matches_independent_ode_including_resonance(self):
        times = np.linspace(0, .002, 101)
        amplitude = 1000.
        for natural, drive in ((0., 10000.), (20000., 10000.), (20000., 20000.), (20000., 20000.0000001)):
            numerical = solve_ivp(lambda t, y: [y[1], amplitude * np.sin(drive * t) - natural**2 * y[0]],
                                  (times[0], times[-1]), [0., 0.], method='DOP853', t_eval=times,
                                  rtol=1e-12, atol=1e-18)
            self.assertTrue(numerical.success)
            np.testing.assert_allclose(sinusoidal_displacement(times, amplitude, drive, natural),
                                       numerical.y[0], rtol=2e-9, atol=1e-15)

    def test_zero_start_and_small_time_jerk_are_resolved(self):
        t = 1e-14
        for natural in (0., 20000.):
            values = sinusoidal_displacement(np.array([0., t]), 1000., 20000., natural)
            self.assertEqual(values[0], 0.)
            self.assertAlmostEqual(values[1] / (1000. * 20000. * t**3 / 6), 1., places=12)
        with self.assertRaises(ValueError):
            sinusoidal_displacement([-1.], 1000., 20000.)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            transient_sif(Material(), dt_s=1e-6, steps=10, drive_omega=20000., probe_node=1)


def probe_fixture(root):
    times = np.arange(1, 5) * 1e-6
    np.savetxt(root / 'probe.dat', np.column_stack((np.arange(4) + 7., times, np.arange(4) + 20.)))
    (root / 'probe.dat.names').write_text('Metadata for solver output\n1: mean: displacement 3\n'
                                         '2: value: time scalar variable\n3: value: displacement 1 at node 2\n')
    return times


class ScalarProbeTests(unittest.TestCase):
    def test_named_columns_override_their_physical_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = probe_fixture(root)
            key = 'value: displacement 1 at node 2'
            result = read_scalar_probe(root, expected_times=times, required_names=[key])
            np.testing.assert_array_equal(result[key], np.arange(4) + 20.)

    def test_wrong_time_missing_probe_and_duplicate_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = probe_fixture(root)
            with self.assertRaisesRegex(ValueError, 'time grid'):
                read_scalar_probe(root, expected_times=times * 2, required_names=[])
            with self.assertRaisesRegex(ValueError, 'missing'):
                read_scalar_probe(root, expected_times=times, required_names=['value: displacement 3 at node 7'])
            path = root / 'probe.dat.names'
            path.write_text(path.read_text().replace('value: displacement 1 at node 2', 'mean: displacement 3'))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_scalar_probe(root, expected_times=times, required_names=[])


@unittest.skipUnless(os.environ.get('TRAXIS_ELMER_HOME'), 'set TRAXIS_ELMER_HOME for the optional transient reference test')
class InstalledTransientElmerTests(unittest.TestCase):
    def test_real_force_equivalence_and_discrete_modal_time_refinement(self):
        with tempfile.TemporaryDirectory() as directory:
            result = verify_elmer_transient(Path(os.environ['TRAXIS_ELMER_HOME']), Path(directory) / 'checked',
                                            ResourceGuard(ResourceLimits(max_seconds=60, max_rss_mib=512)))
            self.assertTrue(result['passed'])
            self.assertTrue(result['completed'])
            self.assertFalse(result['ready_for_training_data'])
            self.assertEqual([row['element_order'] for row in result['orders']], [1, 2])
            self.assertGreater(result['orders'][1]['negative_uniform_nodal_coefficients'], 0)
            self.assertEqual(result['training_updates'], 0)
