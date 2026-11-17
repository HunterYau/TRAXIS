"""Software checks for the five-measurement benchmark, not extra physical trials."""
import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile
import json
from unittest.mock import patch
import numpy as np

from examples.experimental_five import (PLATE, point_row, fbg_row, radial_edge_load,
                                        mesh_screen, arrival)
from traxis.gpu_tensor import make_layout
from traxis.gpu_enriched_verlet import gpu_explicit_simulate, sparse_probe_set
from traxis.storage import write_json
import examples.experimental_five as experiment


class ExperimentalPorts(unittest.TestCase):
    def test_ring_resultant_moments_and_quadrature(self):
        layout=make_layout(PLATE,(8,8,1),4)
        load=radial_edge_load(layout).reshape(3,-1)
        refined=radial_edge_load(layout,48).reshape(3,-1)
        points=layout.points()
        np.testing.assert_allclose(load,refined,rtol=1e-11,atol=1e-14)
        np.testing.assert_allclose(load.sum(axis=1),0.,atol=1e-13)
        moment=load@points
        np.testing.assert_allclose(moment,np.diag([np.pi*.005**2,np.pi*.005**2,0]),atol=1e-13)
        np.testing.assert_allclose(np.cross(points,load.T).sum(axis=0),0.,atol=1e-13)

    def test_fbg_and_point_reproduce_polynomial_fields(self):
        layout=make_layout(PLATE,(8,8,1),4)
        xyz=layout.points(); field=xyz[:,0]**3+2*xyz[:,1]+3*xyz[:,2]
        for z in (0.,.001):
            ids,weights=fbg_row(layout,z)
            expected=(.255**3-.245**3)/.01
            self.assertAlmostEqual(weights@field[ids],expected,places=11)
            ids,weights=point_row(layout,(.251,.499,z))
            self.assertAlmostEqual(weights@field[ids],.251**3+2*.499+3*z,places=12)

    def test_bloch_unfolded_branch_beyond_first_zone(self):
        result=mesh_screen(250,5,250000)
        self.assertLess(abs(result['phase_relative_error']),.001)
        self.assertLess(abs(result['group_relative_error']),.003)

    def test_arrival_rejects_truncated_simulation(self):
        with self.assertRaises(ValueError):
            arrival(np.linspace(0,.0001,100),np.zeros(100),100000)

    def test_large_layout_requires_explicit_dof_and_memory_budgets(self):
        with self.assertRaises(MemoryError):
            make_layout(PLATE,(250,250,1),5)
        layout=make_layout(PLATE,(250,250,1),5,max_dofs=32000000,
                           max_device_mib=6144,max_host_mib=4096)
        self.assertEqual(layout.screen['vector_dofs'],28170018)
        with self.assertRaises(ValueError):
            make_layout(PLATE,(250,250,1),5,max_dofs=32000001)

    def test_multi_gib_workspace_rejected_by_budget_before_gpu_allocation(self):
        # Broadcast arrays exercise >2 GiB accounting without allocating that
        # much host memory. On Windows the old NumPy-int32 addition overflowed.
        n=28170018
        mass=np.broadcast_to(np.array(1.),(n,))
        load=np.broadcast_to(np.array(0.),(n,))
        system=SimpleNamespace(mass_diagonal=mass,stable_dt_s=1e-8)
        probes=sparse_probe_set([(np.array([0]),np.array([1.]))],n)
        with self.assertRaisesRegex(MemoryError,'declared screen'):
            gpu_explicit_simulate(system,16,1e-9,probes,load,max_workspace_mib=1024)

    def test_worker_saves_two_progress_receipts_and_complete_outputs(self):
        # A stub trajectory tests the whole file lifecycle; this is not another
        # simulation or comparison with a physical specimen.
        class Stiffness:
            storage=SimpleNamespace(cp=SimpleNamespace(cuda=SimpleNamespace(
                get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None))))
            def __matmul__(self,state):return np.zeros_like(state)
        def build(layout,material,guard=None):
            return SimpleNamespace(stiffness=Stiffness(),stable_dt_s=1.,screen={},device={})
        def simulate(system,steps,dt,probes,load,**kwargs):
            for step in (16,32):kwargs['progress'](dict(steps_completed=step))
            return SimpleNamespace(time_s=np.linspace(0,.00035,100),displacement_m=np.zeros((100,2)),
                final_displacement_m=np.zeros(len(load)),final_velocity_m_s=np.zeros(len(load)),
                energy_time_s=np.array([.0002,.00035]),modified_energy_j=np.ones(2),physical_energy_j=np.ones(2),statistics={})
        row=dict(nifgen_avg_waveform_ch1=np.ones(2000),sampleRate=5000000,
                 niscope_avg_waveform=np.zeros((2000,4)))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.multiple(experiment,ROOT=root,CELLS=(4,4,1),ORDER=4,STEPS=32), \
                 patch.object(experiment,'record',return_value=row), \
                 patch.object(experiment,'build_gpu_tensor',side_effect=build), \
                 patch.object(experiment,'gpu_explicit_simulate',side_effect=simulate), \
                 patch.object(experiment,'streamed_action',side_effect=lambda layout,mat,state,*a,**kw:np.zeros_like(state)):
                write_json(root/'protocol.json',experiment.protocol())
                experiment.worker(50000,root/'native')
            self.assertEqual(len(list((root/'native').glob('progress-*.json'))),2)
            result=json.loads((root/'native/result.json').read_text())
            self.assertTrue(result['completed'])
            self.assertFalse(result['ready_for_training_data'])
            self.assertTrue((root/'native/waveforms.npz').is_file())
            self.assertTrue((root/'native/final-state.npz').is_file())


if __name__=='__main__':unittest.main()
