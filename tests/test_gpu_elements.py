import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from examples.matrixfree_capacity import measure_capacity
from examples.matrixfree_step import measure_step
from tests.test_matrixfree import capacity_settings
from tests.test_matrixfree_step import step_settings
from traxis.config import Plate
from traxis.geometry import structured_mesh
from traxis.gpu_elements import build_gpu_elements,GPUElementOperator,storage_screen
from traxis.gpu_verification import verify_gpu
from traxis.matrixfree import MatrixFreeSettings
from traxis.resources import ResourceGuard,ResourceLimits


class GPUPreflightTests(unittest.TestCase):
    def test_device_storage_and_chunk_limits_fail_before_native_or_cuda_allocation(self):
        mesh = structured_mesh(Plate(.04,.03,.003),(2,2,1))
        settings = MatrixFreeSettings(order=3)
        screen = storage_screen(mesh,settings)
        self.assertEqual(screen['vector_dofs'],588)
        with patch('traxis.gpu_elements._native_space',side_effect=AssertionError('native allocation forbidden')):
            with self.assertRaises(MemoryError):
                build_gpu_elements(mesh,None,settings,max_device_mib=1e-6)
        for changes in ({'chunk_elements':0},{'device_id':-1},{'max_device_mib':float('nan')}):
            with self.assertRaises(ValueError):
                storage_screen(mesh,settings,**changes)

    def test_fused_operator_preserves_linear_combination_and_array_interfaces(self):
        k = np.arange(36,dtype=float).reshape(6,6)
        k = k+k.T+np.eye(6)*50
        m = np.eye(6)*2+np.ones((6,6))*.1
        calls = []
        def apply(x,ks,ms):
            calls.append((ks,ms))
            return (ks*k+ms*m)@x
        storage = SimpleNamespace(ns=2,apply=apply)
        stiffness,mass = GPUElementOperator(storage,1,0),GPUElementOperator(storage,0,1)
        effective = mass+.25*stiffness
        self.assertIsInstance(effective,GPUElementOperator)
        x = np.arange(6,dtype=float)
        np.testing.assert_allclose(effective@x,(m+.25*k)@x)
        self.assertEqual(calls,[(.25,1.)])
        block = np.column_stack((x,x+1))
        np.testing.assert_allclose((2*mass+.5*stiffness)@block,(2*m+.5*k)@block)
        np.testing.assert_allclose(effective.T@x,(m+.25*k).T@x)
        with self.assertRaises(ValueError):
            stiffness*float('nan')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                verify_gpu(Path(directory)/'uncreated',orders=())
            self.assertFalse((Path(directory)/'uncreated').exists())


@unittest.skipUnless(os.environ.get('TRAXIS_GPU_CHECKS') == '1' and importlib.util.find_spec('cupy')
                     and importlib.util.find_spec('ngsolve'),'enable optional CUDA/native operator checks')
class NativeGPUReferenceTests(unittest.TestCase):
    def test_all_columns_fused_operator_and_transients_match_sparse_references(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'reference'
            report = verify_gpu(output,ResourceGuard(ResourceLimits(max_seconds=120,max_rss_mib=1024)))
            self.assertTrue(report['completed'])
            self.assertTrue(report['passed'])
            self.assertFalse(report['ready_for_training_data'])
            self.assertEqual(len(report['cases']),8)
            for row in report['cases']:
                self.assertEqual(row['all_coefficient_columns_checked'],row['dofs'])
                self.assertTrue(row['fused_effective_operator'])
                self.assertEqual(row['device']['precision'],'float64')
                for key in ('stiffness_relative_frobenius','mass_relative_frobenius','effective_relative_frobenius',
                            'stiffness_diagonal_relative_l2','mass_diagonal_relative_l2','block_application_relative_l2'):
                    self.assertLess(row[key],1e-10)
                self.assertLess(row['maximum_relative_residual'],1e-9)
                self.assertLess(row['postburst_relative_energy_range'],1e-8)
                with np.load(output/'native'/row['comparison_file'],allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays['time_s'],np.arange(33)*2.5e-7)
                    self.assertEqual(arrays['gpu_m'].dtype,np.float64)
                    measured = np.linalg.norm(arrays['gpu_m']-arrays['direct_m'])/np.linalg.norm(arrays['direct_m'])
                    self.assertAlmostEqual(measured,row['waveform_relative_l2'],delta=1e-20)

    def test_streamed_gpu_first_step_matches_direct_native_newmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capacity = measure_capacity({**capacity_settings(),'mesh_size_m':.008},root/'capacity',
                ResourceGuard(ResourceLimits(max_seconds=45,max_rss_mib=1024)))
            self.assertTrue(capacity['completed'],capacity.get('error'))
            cfg = {**step_settings(),'backend':'gpu_elements',
                'gpu':{'max_device_mib':16,'chunk_elements':31,'device_id':0}}
            result = measure_step(root/'capacity',cfg,root/'step',
                ResourceGuard(ResourceLimits(max_seconds=45,max_rss_mib=1024)))
            self.assertTrue(result['completed'],result.get('error'))
            native = json.loads((root/'step/native/step-report.json').read_text())
            self.assertEqual(native['backend'],'gpu_elements')
            self.assertEqual(native['device']['precision'],'float64')
            self.assertLess(native['direct_acceleration_relative_l2'],1e-7)
            self.assertLess(native['relative_energy_balance_error'],1e-8)
            self.assertGreater(native['solve_progress'][-1]['effective_operator_applications'],0)
