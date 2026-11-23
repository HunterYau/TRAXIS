"""Tiny CPU verification only. No real plate simulation, GPU work, or cloud calls."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from examples import pzt_mor_study as study
from examples.pzt_broadband import reconstruct, paired_metrics, within_target, interior_impulse
from examples.pzt_era import fit_era, reduced_impulse, hankel, kernel_data, dimensions
from tests.test_pzt_broadband import simulate
from traxis.storage import sha256_file, write_json


def direct(a, b, c, drive):
    # Independent driven state-space recurrence, rather than convolution.
    state = np.zeros(len(a))
    output = np.zeros((len(drive), len(c)))
    for node in range(2, len(drive)):
        state = a @ state + b * drive[node - 1]
        output[node] = c @ state
    return output


def fixture():
    a = np.array([[.97, -.09, 0., 0.], [.09, .97, 0., 0.],
                  [0., 0., .93, -.21], [0., 0., .21, .93]])
    b = np.array([1., -.3, .7, .4])
    c = np.array([[1., .3, -.2, .5], [1.001, .299, -.199, .501]])
    t = np.arange(129) * .002
    drive = np.zeros(len(t)); drive[1] = 1.
    return t, direct(a, b, c, drive), a, b, c


class EraMathTests(unittest.TestCase):
    def test_block_hankel_interleaving_and_shift(self):
        data = np.column_stack([np.arange(7), 10 + np.arange(7)])
        np.testing.assert_array_equal(hankel(data, 2, 3),
            [[0, 1, 2], [10, 11, 12], [1, 2, 3], [11, 12, 13]])
        np.testing.assert_array_equal(hankel(data, 2, 3, 1),
            [[1, 2, 3], [11, 12, 13], [2, 3, 4], [12, 13, 14]])

    def test_exact_rank_recovery_and_independent_six_drives(self):
        t, h, a, b, c = fixture()
        model, diagnostic = fit_era(t, h, 12, backend='numpy')
        self.assertEqual(diagnostic['available_order'], 4)
        reduced, info = reduced_impulse(model, 4, len(t))
        np.testing.assert_allclose(reduced, h, atol=2e-13, rtol=1e-11)
        self.assertFalse(info['poles_modified'])
        for frequency in (5., 10., 15., 20., 25., 30.):
            drive = np.sin(2 * np.pi * frequency * t) * np.sin(np.pi * t / t[-1])**2
            expected = direct(a, b, c, drive)
            score = paired_metrics(expected, reconstruct(reduced, drive))
            self.assertTrue(within_target(score, 1e-7), score)
        coarse, _ = reduced_impulse(model, 2, len(t))
        self.assertFalse(within_target(paired_metrics(h, coarse), .1))

    def test_production_verlet_consistent_mass_and_cancelling_receivers(self):
        mass = [[2., .4, .1], [.4, 1.5, .2], [.1, .2, 1.]]
        stiffness = [[3100., -900., 300.], [-900., 2400., -350.], [300., -350., 1900.]]
        probes, load = [[.3, 1., -.2], [.3, 1.001, -.199]], [1., -.3, .2]
        n, dt = 128, .001
        full = simulate(mass, stiffness, load, probes, n, dt, interior_impulse(dt))
        model, diagnostic = fit_era(full.time_s, full.displacement_m, 10, backend='numpy')
        self.assertEqual(diagnostic['available_order'], 6)
        reduced, _ = reduced_impulse(model, 6, n + 1)
        rng = np.random.default_rng(5)
        drive = rng.normal(size=n + 1); drive[0] = 0.
        actual = simulate(mass, stiffness, load, probes, n, dt, lambda t: drive[round(t / dt)])
        scores = paired_metrics(actual.displacement_m, reconstruct(reduced, drive))
        self.assertTrue(within_target(scores, 1e-6), scores)

    def test_unstable_models_rejected_without_changing_poles(self):
        model = dict(a=np.array([[1.1]]), b=np.ones((1, 1)), c=np.ones((2, 1)), singular_values=np.ones(1))
        with self.assertRaises(ArithmeticError):
            reduced_impulse(model, 1, 100)
        self.assertEqual(model['a'][0, 0], 1.1)

    def test_complex_malformed_or_nonuniform_inputs_rejected(self):
        t, h, *_ = fixture()
        bad = h.astype(complex); bad[2, 0] += 1j
        with self.assertRaises(ValueError):
            kernel_data(t, bad)
        broken = t.copy(); broken[4] += .0001
        with self.assertRaises(ValueError):
            kernel_data(broken, h)
        model, _ = fit_era(t, h, 4, backend='numpy')
        model['a'] = model['a'].astype(complex) + 1j
        with self.assertRaises(ValueError):
            reduced_impulse(model, 4, len(t))
        with self.assertRaises(ValueError):
            fit_era(t, h, 4, rows=100, columns=100, backend='numpy')

    def test_full_size_estimate_does_not_allocate(self):
        with patch('numpy.empty', side_effect=AssertionError('No matrix allocation')):
            size = dimensions(65537)
        self.assertEqual(size['hankel_shape'], [65534, 32768])
        self.assertLess(size['conservative_device_workspace_bytes'], 113 * 1024**3)


class MorScheduleTests(unittest.TestCase):
    def test_prepare_does_not_compute_or_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'prepared'
            with patch.object(study, 'reference_identity', return_value={'samples': 65537}), \
                 patch.object(study, 'sources', return_value={}), \
                 patch.object(study.queue, 'preflight', return_value={'source_sha256': {}}), \
                 patch.object(study, 'target_source', return_value=(None, {'fixture': True})), \
                 patch.object(study, 'fit_era') as fit, \
                 patch.object(study, 'reduced_impulse') as propagate, \
                 patch.object(study, 'reconstruct') as convolution, \
                 patch.object(study.subprocess, 'Popen') as launch:
                study.prepare(root)
                for operation in (fit, propagate, convolution, launch):
                    operation.assert_not_called()
            plan = json.loads((root / 'plan.json').read_text())
            self.assertEqual(plan['stages'][:2], ['full-order-300khz', 'build-mor'])
            self.assertEqual(plan['frequencies_hz'], [50000, 100000, 150000, 200000, 250000, 300000])
            self.assertEqual(len(plan['orders']), 7)
            self.assertFalse(json.loads((root / 'status.json').read_text())['computation_started'])

    def test_300khz_is_explicitly_synthetic(self):
        pulse, metadata = study.target_source(300000)
        self.assertFalse(metadata['experimental_generator_verified'])
        self.assertFalse(metadata['experimental_receiver_comparison_available'])
        self.assertEqual(pulse(np.array([0., 1e-3])).tolist(), [0., 0.])

    def test_tiny_worker_pipeline_artifacts_and_tamper_detection(self):
        t, h, a, b, c = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = dict(stages=['full-order-300khz', 'build-mor', 'mor-order-0002', 'mor-order-0004'],
                        limitations=['Tiny test fixture; not a real plate'],
                        reference={'original_broadband_solver_seconds': 232.,
                                   'original_five_reconstruction_seconds': .215})
            write_json(root / 'plan.json', plan)

            def source(f):
                frequency = f / 10000
                return (lambda t: np.sin(2 * np.pi * frequency * t) * np.sin(np.pi * t / .256)**2), {'fixture': True}

            def reference(_, f):
                return t, direct(a, b, c, source(f)[0](t))

            def fit(time_s, voltage, maximum, **kwargs):
                return fit_era(time_s, voltage, maximum, backend='numpy')

            with patch.object(study, 'load_plan', return_value=plan), \
                 patch.object(study, 'saved_kernel', return_value=(t, h)), \
                 patch.object(study, 'target_source', side_effect=source), \
                 patch.object(study, 'reference_trace', side_effect=reference), \
                 patch.object(study, 'ORDERS', [2, 4]), \
                 patch.object(study, 'fit_era', side_effect=fit):
                for stage in plan['stages']:
                    study.worker(root, stage)
                result = study.assess(root)
            self.assertTrue(result['completed'])
            self.assertTrue(result['numerical_method_pass'])
            self.assertEqual(result['selected_order'], 4)
            self.assertFalse(result['ready_for_training_data'])
            self.assertFalse(result['timing_comparison']['same_hardware_and_fresh_full_order_rerun'])
            self.assertGreater(result['timing_comparison']['era_build_plus_selected_online_seconds'],
                               result['timing_comparison']['mor_online_seconds'])
            (root / 'mor-order-0004/100khz/waveforms.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                study.assess(root)

    def test_serial_resume_skips_finished_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {'stages': ['full-order-300khz', 'build-mor', 'mor-order-0032']}
            write_json(root / 'plan.json', plan)
            (root / 'findings.txt').write_text('fixture')
            completed = {'full-order-300khz'}
            launched, active = [], set()

            def launch(command, **kwargs):
                self.assertFalse(active)
                stage = command[-1]; launched.append(stage); active.add(stage)
                def wait():
                    active.remove(stage); completed.add(stage)
                    return 0
                return SimpleNamespace(pid=99999999, stdout=io.StringIO('fixture\n'), wait=wait)

            with patch.object(study, 'load_plan', return_value=plan), \
                 patch.object(study, 'receipt', side_effect=lambda folder, _: {} if folder.name in completed else None), \
                 patch.object(study, 'worker_alive', return_value=False), \
                 patch.object(study, 'assess'), \
                 patch.object(study.subprocess, 'Popen', side_effect=launch):
                # Nonempty receipts are required for truth testing in run().
                study.receipt.side_effect = lambda folder, _: {'completed': True} if folder.name in completed else None
                study.run(root)
                study.run(root)
            self.assertEqual(launched, plan['stages'][1:])

    def test_receipt_requires_all_waveforms(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'full-order-300khz'; folder.mkdir()
            (folder / 'source.json').write_text('{}')
            write_json(folder / 'result.json', dict(completed=True, stage=folder.name, plan_sha256='x',
                artifacts_sha256={'source.json': sha256_file(folder / 'source.json')}))
            with self.assertRaises(ValueError):
                study.receipt(folder, 'x')

    def test_zero_reference_reports_unresolved_and_rejected_model_continues(self):
        t, h, *_ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = dict(stages=['full-order-300khz', 'build-mor', 'mor-order-0002', 'mor-order-0004'],
                        limitations=['Tiny fixture only'])
            write_json(root / 'plan.json', plan)
            def fit(t, h, maximum, **kwargs):
                return fit_era(t, h, maximum, backend='numpy')
            def reduced(model, order, count, **kwargs):
                if order == 2:
                    raise ArithmeticError('Fixture unstable candidate; no pole clipping')
                return np.zeros_like(h), {'impulse_seconds': .001}
            with patch.object(study, 'load_plan', return_value=plan), \
                 patch.object(study, 'saved_kernel', return_value=(t, h)), \
                 patch.object(study, 'target_source', return_value=(lambda t: np.zeros_like(t), {})), \
                 patch.object(study, 'reference_trace', return_value=(t, np.zeros_like(h))), \
                 patch.object(study, 'ORDERS', [2, 4]), \
                 patch.object(study, 'fit_era', side_effect=fit), \
                 patch.object(study, 'reduced_impulse', side_effect=reduced):
                for stage in plan['stages']:
                    study.worker(root, stage)
                result = study.assess(root)
            self.assertTrue(result['completed'])
            self.assertFalse(result['numerical_method_pass'])
            self.assertTrue(result['candidate_results'][0]['rejected'])
            self.assertIsNone(result['candidate_results'][1]['selection_error_percent'])

    def test_frozen_source_change_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = dict(source_sha256={'changed': 'old'}, production_preflight={}, reference={},
                        stages=study.stages(), orders=study.ORDERS, frequencies_hz=study.FREQUENCIES,
                        target_percent=study.TARGET, build_backend='cupy_exact_float64_svd')
            write_json(root / 'plan.json', plan)
            with patch.object(study, 'sources', return_value={'changed': 'new'}):
                with self.assertRaises(ValueError):
                    study.load_plan(root)


if __name__ == '__main__':
    unittest.main()
