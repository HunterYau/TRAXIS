"""Reference-channel and reciprocal port checks without time integration."""
import unittest
from unittest.mock import patch

import numpy as np

from examples import wandowski_pzt_queue as queue
from traxis.config import Plate
from traxis.gpu_tensor import make_layout


class PreparedPZTTests(unittest.TestCase):
    def test_measurements_select_pzt_not_optical_channels(self):
        data = np.arange(40.).reshape(10, 4)
        actual = queue.measured_channels({'niscope_avg_waveform': data})
        np.testing.assert_array_equal(actual, data[:, 2:4])
        with self.assertRaises(ValueError):
            queue.measured_channels({'niscope_avg_waveform': data[:, :3]})

    def test_preflight_reads_real_five_records_and_never_builds_gpu(self):
        with patch.object(queue, 'build_gpu_tensor', side_effect=AssertionError('GPU started')):
            result = queue.preflight()
        self.assertEqual([c['frequency_hz'] for c in result['cases']],
                         [50000, 100000, 150000, 200000, 250000])
        self.assertTrue(all(c['measured_samples'] == 12000 for c in result['cases']))
        self.assertFalse(result['simulation_started'])
        self.assertFalse(result['ready_for_training_data'])

    def test_actuator_virtual_work_and_receiver_area_on_small_layout(self):
        config = queue.configuration()
        layout = make_layout(Plate(*config['plate_m']), (4, 4, 1), 2)
        force, _, receivers = queue.ports(layout, config)
        # Affine biaxial strain: independent area formula for both receivers and
        # work supplied by a one-volt reciprocal planar actuator.
        nx, ny, nz = [n*layout.order+1 for n in layout.cells]
        xyz = np.stack(np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny),
                                  np.linspace(0, .001, nz), indexing='ij'))
        u = np.zeros((3, nx, ny, nz)); u[0] = xyz[0]*2e-6; u[1] = xyz[1]*3e-6
        planar = queue.PiezoMaterial.from_profile(queue.read_profile(queue.PROJECT/config['material_profile'])).planar()
        expected = np.pi*config['disk_radius_m']**2*np.dot(planar.coupling_c_m2, [2e-6, 3e-6, 0])
        for receiver in receivers:
            self.assertAlmostEqual(receiver.strain_charge(u.ravel())/expected, 1., places=10)
        self.assertAlmostEqual(float(force@u.ravel())/(-expected), 1., places=10)
        ns = nx*ny*nz
        for axis in range(3):
            self.assertAlmostEqual(float(force[axis*ns:(axis+1)*ns].sum()), 0., places=13)


if __name__ == '__main__':
    unittest.main()
