import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from traxis.config import Material, Plate
from traxis.geometry import structured_mesh
from traxis.highorder import highorder_dofs
from traxis.matrixfree import MatrixFreeSettings, build_matrix_free
from traxis.matrixfree_verification import _cached_case, _checked_case, run_worker, verify_matrix_free
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import canonical_hash, sha256_file, write_json
from examples.matrixfree_capacity import checked_settings, measure_capacity


def capacity_settings():
    return dict(plate=dict(width_m=.04,height_m=.03,thickness_m=.003),material=dict(young_pa=68.9e9,poisson=.33,density_kg_m3=2700.),
        mesh_size_m=.006,max_geometry_mib=16,max_tetrahedra=10000,
        reference=dict(order=2,max_dofs=20000,max_vertices=10000,max_tetrahedra=10000,max_topology_mib=16,max_vector_mib=16),
        seed=2026,operator_repeats=2)


class MatrixFreePreflightTests(unittest.TestCase):
    def test_topology_storage_is_screened_before_large_unique_workspaces(self):
        mesh = structured_mesh(Plate(.04,.03,.003),(2,2,1))
        with patch('traxis.highorder.np.unique',side_effect=AssertionError('must screen first')):
            with self.assertRaisesRegex(MemoryError,'topology workspace'):
                highorder_dofs(mesh,3,max_topology_mib=1e-6)
        counts = highorder_dofs(mesh,3,max_vertices=100000,max_tetrahedra=200000)
        self.assertEqual(counts['vector_dofs'],588)

    def test_space_limits_fail_before_native_allocation(self):
        mesh = structured_mesh(Plate(.04,.03,.003),(2,2,1))
        for changes in ({'order':5},{'max_dofs':0},{'max_vector_mib':0}):
            with self.assertRaises(ValueError):
                MatrixFreeSettings(**changes)
        with patch('traxis.matrixfree._native_space',side_effect=AssertionError('native allocation is forbidden')):
            for settings in (MatrixFreeSettings(max_dofs=12),MatrixFreeSettings(max_vector_mib=1e-6)):
                with self.assertRaises(MemoryError):
                    build_matrix_free(mesh,Material(),settings)

    def test_empty_or_escaped_verification_cases_cannot_count_as_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for plan in ({'orders':[]}, {'orders':[1],'meshes':[{'name':'../escape','cells':[2,2,1]},
                        {'name':'affine-skewed','cells':[2,2,1]}]}):
                path = root/'plan.json'
                path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError,'cases or fixed tolerances'):
                    run_worker(path,root/'absent',ResourceGuard())
                self.assertFalse((root/'absent').exists())
            with self.assertRaises(ValueError):
                verify_matrix_free(root/'absent',orders=())


class MatrixFreeCacheTests(unittest.TestCase):
    def case(self,root):
        name = 'rectangular-p1'
        case = root/name
        case.mkdir()
        times = np.arange(33)*2.5e-7
        arrays = dict(time_s=times,direct_m=times[:,None]*1e-9,iterative_m=times[:,None]*1e-9,
            energy_time_s=times,energy_j=np.r_[0.,np.ones(32)],iterations=np.r_[0,np.full(32,2)].astype(np.int32))
        path = case/'comparison.npz'
        np.savez_compressed(path,**arrays)
        row = dict(geometry='rectangular',order=1,dofs=54,all_coefficient_columns_checked=54,
            stiffness_relative_frobenius=0.,mass_relative_frobenius=0.,mass_diagonal_relative_l2=0.,
            stiffness_diagonal_relative_l2=0.,block_application_relative_l2=0.,waveform_relative_l2=0.,
            postburst_relative_energy_range=0.,maximum_relative_residual=5e-12,total_cg_iterations=64,
            maximum_cg_iterations=2,stiffness_applications=100,mass_applications=100,native_operator_type='BaseMatrix',
            interface='physical circular normal-force/mean-displacement covectors',passed=True,
            comparison_file=name+'/comparison.npz',comparison_sha256=sha256_file(path),cache_hit=False,computation_seconds=.1)
        plan = dict(fingerprint='f'*64,meshes=[dict(name='rectangular',cells=[2,2,1])],operator_tolerance=1e-10,
            waveform_tolerance=1e-7,energy_tolerance=1e-8,iterative_settings=dict(max_iterations=500,residual_limit=1e-9))
        receipt = dict(plan_fingerprint=plan['fingerprint'],case=row)
        receipt['fingerprint'] = canonical_hash(receipt)
        write_json(case/'case-receipt.json',receipt)
        return name,row,plan,arrays

    def test_checked_recovery_copies_complete_arrays_without_native_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name,row,plan,_ = self.case(root)
            output = root/'recovered'
            output.mkdir()
            recovered = _cached_case(root,name,output,plan)
            self.assertTrue(recovered['cache_hit'])
            self.assertEqual((root/row['comparison_file']).read_bytes(),(output/row['comparison_file']).read_bytes())
            _checked_case(recovered,output,name,plan)
            self.assertIsNone(_cached_case(root,'absent',output,plan))

    def test_incomplete_counts_metrics_and_paths_cannot_count_as_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name,row,plan,_ = self.case(root)
            malformed = [{},*({**row,**change} for change in (
                {'all_coefficient_columns_checked':1},{'maximum_relative_residual':float('nan')},
                {'passed':False},{'waveform_relative_l2':.01},{'dofs':55},{'total_cg_iterations':63},
                {'comparison_file':'../comparison.npz'},{'comparison_file':name+'\\comparison.npz'}))]
            for candidate in malformed:
                with self.subTest(candidate=candidate):
                    with self.assertRaises(ValueError):
                        _checked_case(candidate,root,name,plan)
            receipt = json.loads((root/name/'case-receipt.json').read_text())
            receipt['case'] = {}
            receipt['fingerprint'] = canonical_hash({k:v for k,v in receipt.items() if k!='fingerprint'})
            (root/name/'case-receipt.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                _cached_case(root,name,root/'new',plan)
            self.assertFalse((root/'new').exists())

    def test_rehashed_but_malformed_or_oversized_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name,row,plan,arrays = self.case(root)
            path = root/row['comparison_file']
            for change in ({'time_s':np.arange(33)*1e-6},{'iterations':np.full(33,501,dtype=np.int32)},
                           {'direct_m':np.zeros((33,1))},{'energy_j':np.zeros(33)},
                           {'direct_m':np.zeros((200000,1))},{'extra':np.zeros(1)}):
                with self.subTest(change=list(change)):
                    np.savez_compressed(path,**{**arrays,**change})
                    current = {**row,'comparison_sha256':sha256_file(path)}
                    with self.assertRaises(ValueError):
                        _checked_case(current,root,name,plan)


class MatrixFreeCapacityTests(unittest.TestCase):
    def test_invalid_capacity_inputs_fail_before_source_or_native_setup(self):
        settings = capacity_settings()
        checked_settings(settings)
        for changes in ({'operator_repeats':1},{'operator_repeats':True},{'seed':-1},{'max_geometry_mib':0},
                        {'max_tetrahedra':10001},{'mesh_size_m':float('nan')},{'unknown':0}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    checked_settings({**settings,**changes})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'uncreated'
            with self.assertRaises(ValueError):
                measure_capacity({**settings,'operator_repeats':0},output)
            self.assertFalse(output.exists())


@unittest.skipUnless(os.environ.get('TRAXIS_HIGHORDER_CHECKS') == '1' and importlib.util.find_spec('ngsolve'),
                     'enable optional isolated native matrix-free checks')
class NativeMatrixFreeTests(unittest.TestCase):
    def test_capacity_driver_preserves_mesh_and_avoids_sparse_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'capacity'
            result = measure_capacity(capacity_settings(),output,
                ResourceGuard(ResourceLimits(max_seconds=45,max_rss_mib=1024)))
            self.assertTrue(result['completed'],result.get('error'))
            self.assertEqual(result['wave_solves'],0)
            self.assertFalse(result['ready_for_training_data'])
            native = json.loads((output/'native/capacity-report.json').read_text())
            self.assertFalse(native['sparse_reference_assembled'])
            self.assertEqual(len(native['operator_actions']),2)
            self.assertEqual(native['stages'][0]['geometry']['screen_policy'],'geometry_only')
            self.assertLess(native['translation_mass_relative_error'],1e-10)
            self.assertLess(native['normalized_rigid_force'],1e-10)

    def test_all_operator_columns_and_short_direct_transients_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            report = verify_matrix_free(Path(directory)/'check',
                ResourceGuard(ResourceLimits(max_seconds=420,max_rss_mib=1024)))
            self.assertTrue(report['completed'])
            self.assertTrue(report['passed'])
            self.assertEqual(len(report['cases']),8)
            self.assertFalse(report['ready_for_training_data'])
            for case in report['cases']:
                self.assertEqual(case['all_coefficient_columns_checked'],case['dofs'])
                self.assertLess(case['stiffness_relative_frobenius'],1e-10)
                self.assertLess(case['mass_relative_frobenius'],1e-10)
                self.assertLess(case['mass_diagonal_relative_l2'],1e-10)
                self.assertLess(case['stiffness_diagonal_relative_l2'],1e-10)
                self.assertLess(case['waveform_relative_l2'],1e-7)
                self.assertLess(case['maximum_relative_residual'],1e-9)
                self.assertLess(case['postburst_relative_energy_range'],1e-8)
            recovered = verify_matrix_free(Path(directory)/'recovered',
                ResourceGuard(ResourceLimits(max_seconds=30,max_rss_mib=1024)),cache_dir=Path(directory)/'check/native')
            self.assertTrue(recovered['passed'])
            self.assertEqual(recovered['new_case_computations'],0)
            self.assertTrue(all(r['cache_hit'] for r in recovered['cases']))
            for before,after in zip(report['cases'],recovered['cases']):
                self.assertEqual(before['comparison_sha256'],after['comparison_sha256'])
