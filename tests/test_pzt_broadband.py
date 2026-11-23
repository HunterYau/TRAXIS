"""Cheap CPU checks of impulse reconstruction using the actual Verlet function."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import copy
import io

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import lfilter

from examples.pzt_broadband import (interior_impulse, reconstruct, common_grid,
                                    paired_metrics, within_target)
from examples import pzt_broadband_study as study
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.storage import write_json


class NumpyDevice:
    """Only array/probe backend replacement; time stepping is production code.

    This does not validate a GPU kernel. It lets small consistent-mass systems
    exercise the exact production integrator and source-time convention on CPU.
    """
    float64 = np.float64
    int32 = np.int32
    asarray = staticmethod(np.asarray)
    asnumpy = staticmethod(lambda value: np.array(value, copy=True))
    zeros = staticmethod(np.zeros)
    empty = staticmethod(np.empty)
    empty_like = staticmethod(np.empty_like)
    concatenate = staticmethod(np.concatenate)
    subtract = staticmethod(np.subtract)
    sum = staticmethod(np.sum)
    cuda = SimpleNamespace(Device=lambda _: nullcontext(),
        runtime=SimpleNamespace(memGetInfo=lambda: (2**40, 2**40)),
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None))

    @staticmethod
    def RawKernel(code, name, **kwargs):
        if name != 'sample_probes':
            raise ValueError('Unexpected GPU kernel in the CPU test')

        def sample(grid, block, args):
            starts, ids, weights, u, output, count, index = args
            for probe in range(count):
                a, b = starts[probe:probe + 2]
                output[index, probe] = np.dot(weights[a:b], u[ids[a:b]])
        return sample


def small_system(mass, stiffness):
    mass, stiffness = np.asarray(mass, dtype=float), np.asarray(stiffness, dtype=float)
    inverse = np.linalg.inv(mass)
    largest = np.max(np.linalg.eigvals(inverse @ stiffness).real)
    storage = SimpleNamespace(cp=NumpyDevice(), lock=nullcontext(), device_id=0, applications=0)

    def action(u, ku, alpha, beta):
        ku[:] = alpha * (stiffness @ u) + beta * ku
        storage.applications += 1

    def invert(vector):
        vector[:] = inverse @ vector

    def quadratic(vector, *, inverse=False):
        return vector @ (np.linalg.solve(mass, vector) if inverse else mass @ vector)

    storage._launch = action
    return SimpleNamespace(mass_diagonal=np.diag(mass), stable_dt_s=2 / np.sqrt(largest),
        inverse_mass_inplace=invert, mass_quadratic=quadratic,
        stiffness=SimpleNamespace(storage=storage, applications=0),
        screen={'device_storage_screen_mib': 0, 'max_device_mib': 512})


def simulate(mass, stiffness, load, probes, steps, dt, force):
    return gpu_explicit_simulate(small_system(mass, stiffness), steps, dt,
        np.asarray(probes, dtype=float), np.asarray(load, dtype=float), force=force,
        output_stride=1, energy_stride=16)


class BroadbandTests(unittest.TestCase):
    def test_consistent_mass_reciprocal_system_matches_arbitrary_inputs(self):
        mass = np.array([[2., .4, .1], [.4, 1.5, .2], [.1, .2, 1.]])
        stiffness = np.array([[3100., -900., 300.], [-900., 2400., -350.], [300., -350., 1900.]])
        load = [1., -.3, .2]
        # Nearly cancelling channels deliberately make the difference sensitive.
        probes = [[.3, 1., -.2], [.3, 1.001, -.199]]
        n, dt = 512, .0005
        impulse = simulate(mass, stiffness, load, probes, n, dt, interior_impulse(dt))
        t = impulse.time_s
        rng = np.random.default_rng(123)
        sources = [rng.normal(size=n + 1),
                   np.where(t < .1, np.sin(2 * np.pi * 50 * t) * np.sin(np.pi * t / .1)**2, 0.),
                   np.where(t < .05, np.sin(2 * np.pi * 100 * t) * np.sin(np.pi * t / .05)**2, 0.)]
        for samples in sources:
            samples[0] = 0.
            direct = simulate(mass, stiffness, load, probes, n, dt,
                              lambda time: samples[round(time / dt)])
            broad = reconstruct(impulse.displacement_m, samples)
            scores = paired_metrics(direct.displacement_m, broad)
            self.assertTrue(within_target(scores, 1e-7))
            np.testing.assert_allclose(broad, direct.displacement_m, rtol=1e-9, atol=1e-14)

    def test_one_step_shift_and_no_circular_wraparound(self):
        kernel = np.zeros((32, 2))
        kernel[2:, 0] = np.arange(30) + 1
        kernel[2:, 1] = .5
        target = np.zeros(32)
        target[25] = -3.
        result = reconstruct(kernel, target)
        np.testing.assert_allclose(result[:26], 0., atol=1e-13)
        np.testing.assert_allclose(result[26:], -3 * kernel[2:8], atol=1e-13)
        target = np.zeros(32)
        target[1] = 1.
        np.testing.assert_allclose(reconstruct(kernel, target), kernel, atol=1e-13)

    def test_time_refinement_matches_independent_continuous_solution(self):
        mass, stiffness = [[2.]], [[800.]]
        duration, frequency = .4, 15.

        def pulse(t):
            return np.sin(2 * np.pi * frequency * t) * np.sin(np.pi * t / .2)**2 if 0 < t < .2 else 0.

        exact = solve_ivp(lambda t, y: [y[1], (pulse(t) - 800 * y[0]) / 2],
                          [0., duration], [0., 0.], rtol=1e-11, atol=1e-13, dense_output=True,
                          max_step=.0005)
        self.assertTrue(exact.success)
        errors = []
        for n in (256, 512, 1024):
            dt = duration / n
            kernel = simulate(mass, stiffness, [1.], [[1.]], n, dt, interior_impulse(dt))
            source = np.array([pulse(t) for t in kernel.time_s])
            reconstructed = reconstruct(kernel.displacement_m, source)[:, 0]
            direct = simulate(mass, stiffness, [1.], [[1.]], n, dt, pulse)
            np.testing.assert_allclose(reconstructed, direct.displacement_m[:, 0], atol=1e-13)
            errors.append(np.linalg.norm(reconstructed - exact.sol(kernel.time_s)[0]))
        # Norms have twice as many samples: expect ratio 4/sqrt(2), not 4.
        self.assertTrue(all(2.6 < a / b < 3.1 for a, b in zip(errors[:-1], errors[1:])), errors)

    def test_rejects_nonzero_initial_input_and_invalid_kernel(self):
        h = np.zeros((8, 2)); h[2:] = 1.
        source = np.ones(8)
        with self.assertRaises(ValueError):
            reconstruct(h, source)
        source[0] = 0.
        h[1] = 1.
        with self.assertRaises(ValueError):
            reconstruct(h, source)
        with self.assertRaises(ValueError):
            interior_impulse(-1.)

    def test_nested_grids_preserve_polarity_gain_and_timing_errors(self):
        t = np.linspace(0., 1., 17)
        v = np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
        a, b = common_grid(t[::2], v[::2], t, v)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(within_target(paired_metrics(v, 2 * v)))
        self.assertFalse(within_target(paired_metrics(v, -v)))
        self.assertFalse(within_target(paired_metrics(v, np.roll(v, 1, axis=0))))


class StudySafetyTests(unittest.TestCase):
    def test_scheduler_runs_serially_and_skips_completed_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, tasks = study.study_design('benchmark')
            plan = {'tasks': tasks}
            write_json(root / 'plan.json', plan)
            (root / 'findings.txt').write_text('fixture summary', encoding='utf-8')
            finished = {tasks[0]['id']}
            active = set()
            launched = []

            def launch(command, **kwargs):
                self.assertFalse(active, 'Workers must not overlap')
                task_id = command[-1]
                active.add(task_id)
                launched.append(task_id)

                def wait():
                    active.remove(task_id)
                    finished.add(task_id)
                    return 0

                return SimpleNamespace(pid=99999999, stdout=io.StringIO('fixture progress\n'), wait=wait)

            def receipt(folder, digest):
                return {'completed': True} if folder.relative_to(root).as_posix() in finished else None

            with patch.object(study, 'load_plan', return_value=plan), \
                 patch.object(study, 'task_receipt', side_effect=receipt), \
                 patch.object(study, 'worker_alive', return_value=False), \
                 patch.object(study, 'assess'), \
                 patch.object(study.subprocess, 'Popen', side_effect=launch):
                study.run(root)
                self.assertEqual(launched, [task['id'] for task in tasks[1:]])
                study.run(root)
                self.assertEqual(len(launched), 5)
            self.assertEqual(json.loads((root / 'status.json').read_text())['status'], 'completed')
            self.assertFalse(active)

    def test_workers_publish_both_methods_and_assessment_end_to_end(self):
        # Exercise file placement, checksum receipts, real experimental
        # comparisons, and timing aggregation with a tiny LTI fixture backend.
        # This is NOT a plate simulation or GPU validation.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = [dict(id=f'baseline/direct/{f // 1000}khz', case='baseline',
                          method='direct', frequency_hz=f) for f in study.FREQUENCIES]
            tasks.append(dict(id='baseline/broadband', case='baseline', method='broadband', frequency_hz=None))
            case = dict(name='baseline', cells=250, body=[32, 6, 3], steps=8192)
            config = copy.deepcopy(study.queue.configuration())
            plan = dict(tasks=tasks, cases=[case], configuration=config, target_percent=1.,
                        suite='benchmark', limitations=['test fixture only'])
            write_json(root / 'plan.json', plan)
            guard = SimpleNamespace(peaks={}, check=lambda **kwargs: None)

            def fixture(system, steps, dt, probes, load, *, force, output_stride, **kwargs):
                self.assertEqual(output_stride, 1)
                t = np.arange(steps + 1) * dt
                values = np.array([force(x) for x in t])
                a = lfilter([0., dt**2], [1., -1.999, .9991], values)
                b = .99 * a + .001 * lfilter([0., dt**2], [1., -1.998, .999], values)
                return SimpleNamespace(time_s=t, displacement_m=np.column_stack([a, b]),
                    energy_time_s=t[::256], modified_energy_j=np.ones(len(t[::256])), statistics={})

            with patch.object(study, 'load_plan', return_value=plan), \
                 patch.object(study.queue, 'build', return_value=(SimpleNamespace(stable_dt_s=1e-6,
                     body_attachments=[]), None, None, 8192)), \
                 patch.object(study, 'PeakGuard', return_value=guard), \
                 patch.object(study, 'ThermalPause'), \
                 patch.object(study, 'gpu_explicit_simulate', side_effect=fixture):
                for task in tasks:
                    study.worker(root, task['id'])
            answer = study.assess(root)
            self.assertTrue(answer['completed'])
            self.assertTrue(answer['all_method_agreements_within_target'])
            self.assertFalse(answer['convergence_evaluated'])
            self.assertFalse(answer['ready_for_training_data'])
            self.assertEqual(len(answer['method_agreement']), 5)
            self.assertTrue(answer['timings'][0]['five_frequency_benchmark'])
            child = root / 'baseline/broadband/50khz/waveforms.npz'
            self.assertTrue(child.is_file())
            child.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                study.assess(root)

    def test_prepare_only_and_expected_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'study'
            fake_config = {'body': {'bond': 'perfect'}}
            with patch.object(study.queue, 'configuration', return_value=fake_config), \
                 patch.object(study, 'identity', return_value={'production_preflight': {'source_sha256': {}}, 'study_sources': {}}), \
                 patch.object(study, 'source', return_value=(None, {'checked': True})), \
                 patch.object(study.subprocess, 'Popen') as launch:
                study.prepare(root, 'convergence')
                launch.assert_not_called()
            plan = json.loads((root / 'plan.json').read_text())
            self.assertEqual(len(plan['tasks']), 39)
            self.assertEqual(sum(t['method'] == 'broadband' for t in plan['tasks']), 12)
            self.assertEqual(plan['output_stride'], 1)
            self.assertFalse(json.loads((root / 'status.json').read_text())['simulation_running'])
            self.assertFalse(json.loads((root / 'findings.json').read_text())['completed'])

    def test_completed_receipt_cannot_skip_missing_waveforms(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_json(folder / 'result.json', dict(completed=True, plan_sha256='frozen',
                       task={'method': 'direct'}, artifacts_sha256={}))
            with self.assertRaises(ValueError):
                study.task_receipt(folder, 'frozen')
