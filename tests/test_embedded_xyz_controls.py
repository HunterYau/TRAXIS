"""Frozen coupled-grid settings, resource envelopes and checkout selection."""
from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from examples.embedded_wave import FULL_LEVELS,analysis_paths,case_config,checked_case,gpu_controls,mesh_controls,screen,validate_plan
from examples.embedded_wave_budget import check_execution,execution_profile,worker_command
from examples.embedded_xyz_preflight import PROPOSAL_SHA256,load_proposal
from examples.tensor_wave import PROJECT,sealed
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import sha256_file


class XYZControlTests(unittest.TestCase):
    def test_all_six_cases_match_the_frozen_proposal_and_cpu_capacity_counts(self):
        proposal=load_proposal()
        counts={False:[4507515,10738611,20352675],True:[4462575,10601499,20062335]}
        for defective in (False,True):
            for i,expected in enumerate(counts[defective]):
                c=case_config('xyz_refinement',xyz_level=i,defective=defective)
                self.assertEqual(c,checked_case(c))
                for key in ('plate','recess','order','geometry_order','steps','duration_s','source_center_xy_m',
                            'receiver_centers_xy_m','radius_m','frequency_hz','cycles','force_scale_n'):
                    self.assertEqual(c[key],proposal[key])
                self.assertEqual({k:c[k] for k in proposal['levels'][i]},proposal['levels'][i])
                self.assertEqual(c['xyz_proposal_sha256'],PROPOSAL_SHA256)
                self.assertEqual(screen(c)['vector_dofs'],expected)
                self.assertEqual(mesh_controls(c)['max_vector_dofs'],24000000)
                self.assertEqual(gpu_controls(c)['max_vector_dofs'],24000000)
                self.assertLess(screen(c)['allocation_upper_bytes'],1024*1024**2)
                samples=(c['steps']+1)*3*8+(c['steps']//c['energy_stride']+1)*24
                self.assertLess((16*expected*8+samples+100000*16)/1024**2,c['max_workspace_mib'])

    def test_changed_levels_physics_grids_and_budgets_reject(self):
        for level in (None,True,1.,-1,3):
            with self.assertRaises(ValueError):case_config('xyz_refinement',xyz_level=level)
        for kwargs in (dict(order=5),dict(steps=294912),dict(h_level=1)):
            with self.assertRaises(ValueError):case_config('xyz_refinement',xyz_level=1,**kwargs)
        for scope in ('smoke','full','h_refinement'):
            with self.assertRaises(ValueError):case_config(scope,xyz_level=1)
        original=case_config('xyz_refinement',xyz_level=2)
        for key,value in [('layers',[1,1]),('steps',294912),('outer_cell_size_m',.005),
                          ('radius_m',.004),('max_workspace_mib',4096),('xyz_proposal_sha256','changed')]:
            c=deepcopy(original);c[key]=value
            with self.assertRaises(ValueError):checked_case(c)

    def test_protocol_and_loader_are_mandatory_snapshot_inputs(self):
        c=case_config('xyz_refinement',xyz_level=0)
        expected=analysis_paths(c,True)
        self.assertEqual(len(expected),5)
        for missing in ('examples/embedded_xyz_preflight.py','configs/embedded-xyz-proposal.json'):
            plan=sealed(dict(kind='embedded_mechanical_window',case=c,mesh_screen=screen(c),
                declared_full_levels=[list(x) for x in FULL_LEVELS],execution={},
                analysis_sources={name:sha256_file(PROJECT/name) for name in expected if name!=missing}))
            with patch('examples.embedded_wave.gpu_package_identity',side_effect=AssertionError('runtime lookup occurred')):
                with self.assertRaises(ValueError):validate_plan(plan,None)
        self.assertEqual(len(analysis_paths(case_config('full',4,294912),True)),3)

    def test_xyz_limits_are_bound_to_the_actual_guard(self):
        c=case_config('xyz_refinement',xyz_level=2);profile=execution_profile(c)
        self.assertEqual(profile['limits']['max_seconds'],28800)
        self.assertEqual(profile['limits']['max_rss_mib'],4096)
        self.assertEqual(profile['limits']['max_system_memory_percent'],89.9)
        self.assertEqual(profile['limits']['max_gpu_memory_percent'],90)
        self.assertEqual(profile['limits']['max_gpu_temperature_c'],80)
        plan=dict(case=c,execution=profile,analysis_sources={'examples/embedded_wave_budget.py':sha256_file(PROJECT/'examples/embedded_wave_budget.py')})
        check_execution(plan,ResourceGuard(ResourceLimits(**profile['limits'])))
        plan['execution']['limits']['max_system_memory_percent']=95
        with self.assertRaises(ValueError):check_execution(plan,ResourceGuard(ResourceLimits(**profile['limits'])))

    def test_child_uses_this_checkout_even_from_another_working_directory(self):
        command=worker_command('unused-plan','unused-output')+['--help']
        result=subprocess.run(command,cwd=PROJECT.parents[1],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--xyz-level',result.stdout)

    def test_cli_rejects_combined_refinement_axes_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'untouched'
            command=worker_command('unused','unused')[:3]+['--h-level','0','--xyz-level','0','--output',str(out)]
            result=subprocess.run(command,cwd=PROJECT.parents[1],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,2)
            self.assertFalse(out.exists())


if __name__=='__main__':
    unittest.main()
