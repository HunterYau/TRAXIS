import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import eigh

from traxis.config import Material,Plate
from traxis.enriched_factor import gradient_factor,local_stability_data,streamed_action
from traxis.enriched_tet import WEIGHTS,assemble_enriched,element_matrices,reference_integrals
from traxis.geometry import structured_mesh
from traxis.gpu_enriched import build_gpu_enriched
from traxis.matrixfree import MatrixFreeSettings
from traxis.resources import ResourceGuard,ResourceLimits


class FactorizedEnrichedTests(unittest.TestCase):
    def test_reference_gradient_factor_and_local_cfl_bound_preserve_all_coefficients(self):
        factor,error = gradient_factor()
        self.assertEqual(factor.shape,(13,3,15))
        self.assertLess(error,1e-11)
        _,gram = reference_integrals()
        np.testing.assert_allclose(factor.reshape(13,45).T@factor.reshape(13,45),
            gram.transpose(0,2,1,3).reshape(45,45),rtol=1e-10,atol=1e-12)
        vertices = np.array([[.04,0.,0.],[.002,.03,.001],[.001,.001,.004],[0.,0.,0.]])
        jac = (vertices[:3]-vertices[3]).T
        material = Material()
        upper,diagonal = local_stability_data(np.linalg.inv(jac)[None],material)
        k,_,mass = element_matrices(vertices,material)
        maximum = eigh(k,np.diag(np.tile(mass,3)),eigvals_only=True)[-1]
        self.assertGreaterEqual(upper[0],maximum*(1-1e-12))
        np.testing.assert_allclose(diagonal[0]*abs(np.linalg.det(jac)),k.diagonal(),rtol=1e-11,atol=1e-7)

    def test_streamed_action_matches_assembled_on_skewed_cells(self):
        mesh = structured_mesh(Plate(.04,.03,.003),(1,1,1))
        mesh.points_m = mesh.points_m@np.array([[1.,.1,0.],[.04,.9,.01],[.01,0.,1.1]]).T
        material = Material(71e9,.29,2800.)
        system = assemble_enriched(mesh,material)
        values = np.random.default_rng(2026).normal(size=system.ndof)
        result = streamed_action(system.topology,material,values,chunk_elements=3)
        self.assertLess(np.linalg.norm(result-system.stiffness@values)/np.linalg.norm(result),1e-11)
        with self.assertRaises(ValueError):
            streamed_action(system.topology,material,values,chunk_elements=0)

    def test_device_budget_fails_before_native_or_cuda_allocation(self):
        mesh = structured_mesh(Plate(.04,.03,.003),(1,1,1))
        with patch('traxis.gpu_enriched._native_space',side_effect=AssertionError('native allocation forbidden')):
            with self.assertRaises(MemoryError):
                build_gpu_enriched(mesh,Material(),max_device_mib=1e-8)
        with self.assertRaises(ValueError):
            build_gpu_enriched(mesh,Material(),chunk_elements=0)


@unittest.skipUnless(os.environ.get('TRAXIS_GPU_CHECKS')=='1' and importlib.util.find_spec('cupy')
    and importlib.util.find_spec('ngsolve'),'enable optional isolated enriched GPU checks')
class NativeFactorizedGPUChecks(unittest.TestCase):
    def test_all_columns_mass_map_and_stability_match_independent_references(self):
        from traxis.gpu_enriched_verification import verify_gpu_enriched
        with tempfile.TemporaryDirectory() as directory:
            result = verify_gpu_enriched(Path(directory)/'reference',
                ResourceGuard(ResourceLimits(max_seconds=90,max_rss_mib=1024)))
            self.assertTrue(result['completed'])
            self.assertTrue(result['passed'])
            self.assertEqual(len(result['cases']),2)
            for row in result['cases']:
                self.assertEqual(row['all_columns_checked'],row['dofs'])
                self.assertLess(row['stiffness_relative_frobenius'],1e-10)
                self.assertLess(row['mass_relative_l2'],1e-12)
                self.assertTrue(row['native_topology_map_verified'])
                self.assertTrue(row['stable_dt_is_conservative'])
