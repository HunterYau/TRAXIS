from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_forward import local_scenario
from traxis.forward import ForwardSettings, generate_waveform_bank
from traxis.resources import ResourceLimitError
from traxis.waveform_checks import WaveformSweepSettings, plot_waveform_sweep, run_waveform_sweep, trace_discrepancies


class TraceComparisonTests(unittest.TestCase):
    def test_gain_and_phase_are_measured_without_alignment_or_rescaling(self):
        times = np.arange(1000) * 1e-6
        phase = .3
        reference = np.cos(2 * np.pi * 100000 * times)[None]
        candidate = 2 * np.cos(2 * np.pi * 100000 * times + phase)[None]
        unchanged = candidate.copy()
        result = trace_discrepancies(candidate, reference, times, [100000.])
        row = result["rows"][0]
        self.assertAlmostEqual(row["relative_l2"], math.sqrt(5 - 4 * math.cos(phase)), places=12)
        self.assertAlmostEqual(row["relative_rms_error"], 1, places=12)
        self.assertAlmostEqual(row["carrier_phase_difference_rad"], phase, places=12)
        self.assertAlmostEqual(row["envelope_relative_l2"], 1, places=12)
        self.assertIsNone(row["envelope_peak_shift_s"])
        np.testing.assert_array_equal(candidate, unchanged)

    def test_packet_peak_delay_and_zero_reference_are_explicit(self):
        times = np.arange(1000) * 1e-6
        reference = (np.exp(-.5 * ((times - 200e-6) / 30e-6)**2) * np.cos(2 * np.pi * 100000 * times))[None]
        result = trace_discrepancies(np.roll(reference, 5, axis=1), reference, times, [100000.])
        self.assertAlmostEqual(result["rows"][0]["envelope_peak_shift_s"], 5e-6, places=12)
        zeros = trace_discrepancies(np.zeros_like(reference), np.zeros_like(reference), times, [100000.])
        self.assertFalse(zeros["all_records_resolved"])
        self.assertIsNone(zeros["maximum_relative_l2"])

    def test_invalid_axes_grids_noise_and_budgets_fail(self):
        times = np.arange(16) * 1e-6
        with self.assertRaises(ValueError):
            trace_discrepancies(np.zeros((1, 15)), np.zeros((1, 16)), times, [100000.])
        with self.assertRaises(ValueError):
            trace_discrepancies(np.zeros((1, 16)), np.zeros((1, 16)), times[::-1], [100000.])
        with self.assertRaises(ValueError):
            WaveformSweepSettings(mesh_sizes_m=(.003, .004, .002))
        with self.assertRaises(ValueError):
            WaveformSweepSettings(time_refinements=(1, 2, 3))
        with tempfile.TemporaryDirectory() as directory:
            noisy = local_scenario(intact=True)
            noisy = replace(noisy, nuisance=replace(noisy.nuisance, relative_noise_std=.01))
            with self.assertRaisesRegex(ValueError, "noise"):
                run_waveform_sweep(noisy, WaveformSweepSettings(), Path(directory) / "noisy")
            with self.assertRaisesRegex(ValueError, "solve limit"):
                run_waveform_sweep(local_scenario(intact=True), WaveformSweepSettings(max_total_solves=1), Path(directory) / "too-many")


class TimeGridTests(unittest.TestCase):
    def test_exact_halving_fixed_grid_and_stability_rejection(self):
        scenario = local_scenario(intact=True)
        base = ForwardSettings(mesh_backend="structured")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "groups"
            _, original = generate_waveform_bank(scenario, base, cache_dir=cache)
            refined, fine = generate_waveform_bank(scenario, replace(base, time_step_refinement=2), cache_dir=cache)
            self.assertEqual(fine["substeps_per_output"], 2 * original["substeps_per_output"])
            self.assertEqual(fine["dt_internal_s"], original["dt_internal_s"] / 2)
            exact, reused = generate_waveform_bank(scenario, replace(base, fixed_substeps_per_output=fine["substeps_per_output"]), cache_dir=cache)
            self.assertEqual(reused["forward_solves"], 0)
            np.testing.assert_array_equal(exact.waveforms, refined.waveforms)
            with self.assertRaisesRegex(ValueError, "stability"):
                generate_waveform_bank(scenario, replace(base, fixed_substeps_per_output=1))


HAS_GMSH = importlib.util.find_spec("gmsh") is not None


@unittest.skipUnless(HAS_GMSH, "optional Gmsh is required")
class RefinementWorkflowTests(unittest.TestCase):
    def test_cache_recovery_independent_axes_and_authenticated_figures(self):
        settings = WaveformSweepSettings(mesh_sizes_m=(.004, .0035, .003),
                                           forward=ForwardSettings(mesh_algorithm=10, mesh_optimization="Netgen", max_internal_steps=50000))
        scenario = local_scenario(intact=True)
        original = generate_waveform_bank
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ResourceLimitError("controlled sweep interruption")
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "shared-groups"
            with patch("traxis.waveform_checks.generate_waveform_bank", interrupted):
                with self.assertRaises(ResourceLimitError):
                    run_waveform_sweep(scenario, settings, root / "interrupted", cache_dir=cache)
            result = run_waveform_sweep(scenario, settings, root / "resumed", cache_dir=cache)
            self.assertTrue(result["completed"])
            self.assertFalse(result["ready_for_training_data"])
            self.assertEqual(len(result["cases"]), 5)
            self.assertEqual(result["cases"][0]["forward_solves"], 0)
            self.assertEqual(result["cases"][0]["cache_hits"], 7)
            self.assertEqual(len({row["dt_internal_s"] for row in result["cases"][:3]}), 1)
            self.assertEqual(result["cases"][3]["dt_internal_s"] / 2, result["cases"][4]["dt_internal_s"])
            self.assertEqual(result["cases"][4]["dt_internal_s"] / 2, result["cases"][2]["dt_internal_s"])
            if importlib.util.find_spec("matplotlib") is not None:
                figures = plot_waveform_sweep(root / "resumed", root / "figures", (0,))
                self.assertEqual(len(figures["files"]), 2)
                path = root / "resumed" / "case-00" / "solver-traces.npz"
                path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    plot_waveform_sweep(root / "resumed", root / "bad-figures", (0,))
