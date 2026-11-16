"""Preserve frozen controls while adding the declared finer-cell cases."""
from copy import deepcopy
from pathlib import Path
import unittest

from examples.embedded_wave import H_LEVELS,case_config,checked_case,mesh_controls,screen
from examples.embedded_wave_budget import check_execution,execution_profile
from examples.tensor_wave import PROJECT,read_json
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import canonical_hash,sha256_file


class HControlTests(unittest.TestCase):
    def test_all_fourteen_legacy_cases_screens_and_profiles_keep_their_identities(self):
        fixture=read_json(Path(__file__).parent/'fixtures/embedded-legacy-identities.json')
        self.assertEqual(len(fixture['cases']),14)
        for row in fixture['cases']:
            c=case_config(row['scope'],row['order'],row['steps'],row['defective'])
            self.assertEqual(canonical_hash(c),row['case_sha256'])
            self.assertEqual(canonical_hash(screen(c)),row['screen_sha256'])
            if row['scope']=='full':self.assertEqual(canonical_hash(execution_profile(c)),row['execution_sha256'])

    def test_h_levels_match_the_predeclared_physical_protocol_and_mesh_counts(self):
        p=read_json(PROJECT/'configs/embedded-h-protocol.json')
        for defective,counts in [(False,[4507515,8090091,12567987]),(True,[4462575,8005599,12431655])]:
            for i,(level,count) in enumerate(zip(p['levels'],counts)):
                c=case_config('h_refinement',4,294912,defective,h_level=i)
                self.assertEqual(checked_case(c),c)
                for key in ('order','geometry_order','layers','steps','duration_s','plate','recess',
                            'source_center_xy_m','receiver_centers_xy_m','radius_m','frequency_hz','cycles','force_scale_n'):
                    self.assertEqual(c[key],p[key])
                self.assertEqual({k:c[k] for k in level},level)
                self.assertEqual(screen(c)['vector_dofs'],count)
                self.assertEqual(mesh_controls(c)['max_vector_dofs'],24000000)
                self.assertLess(screen(c)['allocation_upper_bytes'],512*1024**2)
        self.assertEqual(len(H_LEVELS),3)
        self.assertEqual(p['finest_relative_change_limit'],.01)

    def test_unknown_levels_altered_physics_and_changed_budgets_reject(self):
        for level in (None,True,1.,-1,3):
            with self.assertRaises(ValueError):case_config('h_refinement',h_level=level)
        for kwargs in (dict(order=5,h_level=1),dict(steps=196608,h_level=1)):
            with self.assertRaises(ValueError):case_config('h_refinement',**kwargs)
        with self.assertRaises(ValueError):case_config('full',4,294912,h_level=0)
        c=case_config('h_refinement',h_level=2)
        for key,value in [('h_mesh_vector_dof_limit',48000000),('outer_cell_size_m',.002),
                          ('radius_m',.004),('geometry_order',4),('max_workspace_mib',4096)]:
            changed=deepcopy(c);changed[key]=value
            with self.assertRaises(ValueError):checked_case(changed)
        profile=execution_profile(c)
        self.assertEqual(profile['limits']['max_system_memory_percent'],89.9)
        self.assertEqual(profile['limits']['max_gpu_memory_percent'],90)
        self.assertEqual(profile['limits']['max_gpu_temperature_c'],80)
        self.assertEqual(profile['limits']['max_rss_mib'],4096)
        self.assertEqual(profile['limits']['max_seconds'],10800)
        plan=dict(case=c,execution=profile,analysis_sources={'examples/embedded_wave_budget.py':sha256_file(PROJECT/'examples/embedded_wave_budget.py')})
        guard=ResourceGuard(ResourceLimits(**profile['limits']));check_execution(plan,guard)
        plan['execution']['limits']['max_system_memory_percent']=95
        with self.assertRaises(ValueError):check_execution(plan,guard)


if __name__=='__main__':unittest.main()
