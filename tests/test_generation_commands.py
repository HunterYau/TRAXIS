import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from traxis.bank import WaveformBank
from traxis.scenarios import load_scenario_plan


class GenerationCommandTests(unittest.TestCase):
    def invoke(self, args):
        from traxis.cli import main
        output = io.StringIO()
        with patch("sys.argv", ["traxis", *map(str, args)]), contextlib.redirect_stdout(output):
            main()
        return json.loads(output.getvalue())

    def test_recipe_variants_export_and_small_local_bank_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            result = self.invoke(["plan-scenarios", "--config", "configs/scenarios-pilot.json", "--variants", 2, "--output", plan])
            self.assertEqual(result["rows"], 4)
            self.assertEqual(result["parents"], 2)
            scenarios = load_scenario_plan(plan)
            self.assertEqual(scenarios[0].parent_specimen_id, scenarios[1].parent_specimen_id)
            exported = self.invoke(["export-solver-request", "--plan", plan, "--index", 2, "--output", root / "request"])
            self.assertEqual(exported["records"], 28)
            settings = root / "forward.json"
            settings.write_text(json.dumps({"mesh_backend": "structured", "cells": [8, 8, 4]}))
            produced = self.invoke(["generate-bank", "--plan", plan, "--index", 0, "--settings", settings,
                                    "--cache-dir", root / "cache", "--output", root / "bank.npz"])
            self.assertEqual(produced["forward_solves"], 7)
            self.assertEqual(produced["validation_status"], "plumbing_only")
            bank = WaveformBank.load(root / "bank.npz")
            self.assertFalse(bank.depth_mm.any())

    def test_highorder_waveform_command_keeps_current_state_frequency_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root/'plan.json'
            self.invoke(['plan-scenarios', '--config', 'configs/highorder-recess-control.json', '--output', plan])
            result = {'completed': True, 'ready_for_training_data': False, 'forward_solves': 1,
                'cache_hit': False, 'dofs': 123, 'state': 'current', 'dt_internal_s': 1e-8, 'wall_seconds': 0.}
            with patch('traxis.highorder_waveforms.run_highorder_waveform_case', return_value=(None, result)) as run:
                observed = self.invoke(['highorder-waveforms', '--plan', plan, '--settings', 'configs/highorder-cad-waveform.json',
                    '--state', 'current', '--frequency-khz', 100, '--cache-dir', root/'cache', '--output', root/'case'])
            self.assertEqual(observed, result)
            self.assertEqual(run.call_args.kwargs['state'], 'current')
            self.assertEqual(run.call_args.kwargs['frequency_hz'], 100000.)
            self.assertEqual(run.call_args.kwargs['cache_dir'], root/'cache')
            self.assertEqual(run.call_args.args[1].mesh_backend, 'cad')
            self.assertEqual(len(run.call_args.args[0].recesses), 1)
