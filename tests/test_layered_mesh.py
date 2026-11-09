import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from examples.layered_mesh import INPUTS,bundle_context,checked_spec,load_bundle_mesh,make_geometry
from examples.enriched_gpu_wave import load_context_mesh,mesh_context
from traxis.config import Plate
from traxis.enriched_tet import build_topology
from traxis.geometry import structured_mesh
from traxis.storage import sha256_file


def specification(cells=(2,3,2)):
    return dict(plate=dict(width_m=.04,height_m=.03,thickness_m=.003),
        material=dict(young_pa=68.9e9,poisson=.33,density_kg_m3=2700.),cells=list(cells),
        max_geometry_mib=128,reference=dict(order=2,max_dofs=20000,max_vertices=2000,max_tetrahedra=4000,
            max_topology_mib=128,max_vector_mib=64))


class LayeredGeometryTests(unittest.TestCase):
    def test_vectorized_geometry_and_counts_match_independent_legacy_construction(self):
        for cells in ((1,1,1),(2,3,2),(3,2,4)):
            with self.subTest(cells=cells):
                values = specification(cells)
                mesh,screen,volume = make_geometry(values)
                old = structured_mesh(Plate(**values['plate']),cells)
                np.testing.assert_array_equal(mesh.points_m,old.points_m)
                np.testing.assert_array_equal(mesh.tetrahedra,old.tetrahedra)
                topology = build_topology(mesh)
                for key in ('vertices','edges','faces','tetrahedra','scalar_dofs','vector_dofs'):
                    self.assertEqual(screen[key],topology.screen[key])
                self.assertAlmostEqual(volume/(.04*.03*.003),1.,places=13)

    def test_limits_and_wrong_source_kind_fail_before_coordinate_allocation(self):
        values = specification((2000,2000,2000))
        with patch('examples.layered_mesh.np.meshgrid',side_effect=AssertionError('allocation forbidden')):
            with self.assertRaises(MemoryError):
                make_geometry(values)
        for change in ({'cells':[1,0,1]},{'cells':[True,1,1]},{'max_geometry_mib':float('nan')},
                       {'unknown':0},{'reference':{**specification()['reference'],'order':3}}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    checked_spec({**specification(),**change})
        for plan in ({'input_sha256':{}},{'capacity_directory':'a','mesh_bundle_directory':'b','input_sha256':{}}):
            with self.assertRaises(ValueError):
                mesh_context(plan)

    def test_bundle_receipts_reconstruction_and_tampered_mesh_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root/'spec.json'
            spec.write_text(json.dumps(specification((2,2,1))))
            bundle = root/'bundle'
            result = subprocess.run([sys.executable,'-m','examples.layered_mesh','--spec',str(spec),'--output',str(bundle)],
                capture_output=True,text=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            context = bundle_context(bundle)
            self.assertFalse(context[1]['native_space_constructed'])
            self.assertFalse(context[1]['cuda_allocation'])
            expected,_,_ = make_geometry(specification((2,2,1)))
            loaded = load_bundle_mesh(bundle,context)
            np.testing.assert_array_equal(loaded.points_m,expected.points_m)
            plan = dict(mesh_bundle_directory=str(bundle),input_sha256={n:sha256_file(bundle/n) for n in INPUTS})
            source,context,layered = mesh_context(plan)
            np.testing.assert_array_equal(load_context_mesh(source,context,layered).tetrahedra,expected.tetrahedra)
            np.savez_compressed(bundle/'geometry/mesh.npz',points_m=loaded.points_m*.9,tetrahedra=loaded.tetrahedra)
            with self.assertRaises(ValueError):
                bundle_context(bundle)


@unittest.skipUnless(os.environ.get('TRAXIS_GPU_CHECKS')=='1' and importlib.util.find_spec('cupy')
    and importlib.util.find_spec('ngsolve'),'enable native GPU layered-input checks')
class NativeLayeredInputTests(unittest.TestCase):
    def test_bundle_to_full_window_and_independent_cpu_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root/'spec.json'
            spec.write_text(json.dumps(specification((2,2,1))))
            settings = json.loads((Path(__file__).resolve().parents[1]/'configs/enriched-gpu-wave.json').read_text())
            settings.update(source_center_xy_m=[.01,.008],receiver_centers_xy_m=[[.03,.022],[.02,.015]])
            config = root/'wave.json'
            config.write_text(json.dumps(settings))
            commands = [
                ['-m','examples.layered_mesh','--spec',str(spec),'--output',str(root/'bundle')],
                ['-m','examples.enriched_gpu_wave','--mesh-bundle',str(root/'bundle'),'--settings',str(config),
                    '--mode','wave','--output',str(root/'wave'),'--max-seconds','90']]
            for command in commands:
                result = subprocess.run([sys.executable,*command],capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            native = json.loads((root/'wave/native/wave-report.json').read_text())
            self.assertTrue(native['full_window_completed'])
            self.assertEqual(native['requested_duration_s'],.0005)
            self.assertLess(native['operator']['relative_l2'],1e-10)
            result = subprocess.run([sys.executable,'-m','examples.verify_enriched_window','--run',str(root/'wave'),
                '--steps',str(native['steps_completed']),'--output',str(root/'cpu')],capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertTrue(json.loads((root/'cpu/cpu-window-check.json').read_text())['passed'])
