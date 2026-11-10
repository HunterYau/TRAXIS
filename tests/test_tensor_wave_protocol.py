"""Physical controls, raw convergence criteria and independent saved-state math."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from examples.tensor_wave import archive_schema,checked_case,read_json,smoke_case
from examples.tensor_wave_study import checked_protocol,case_for,case_levels,compare_raw
from examples.verify_tensor_window import cpu_state
from traxis.config import Material,Plate
from traxis.gpu_tensor import make_layout
from traxis.storage import sha256_file
from traxis.tensor_hex import assemble_tensor,make_grid


class TensorWaveProtocolTests(unittest.TestCase):
    def protocol(self):return read_json('configs/tensor-wave-protocol.json')

    def test_fixed_controls_and_five_unique_runs(self):
        cfg=checked_protocol(self.protocol())
        self.assertEqual(len(case_levels(cfg)),5)
        for p,n in case_levels(cfg):checked_case(case_for(cfg,p,n))
        for name,value in (('duration_s',.00025),('spatial_finest_change_limit',.1),('time_steps',[32768,65536]),
                ('receiver_centers_xy_m',[[.1,.1],[.2,.2],[.3,.3]])):
            changed=deepcopy(cfg);changed[name]=value
            with self.assertRaises(ValueError):checked_protocol(changed)
        case=smoke_case();case['steps']=64
        with self.assertRaises(ValueError):checked_case(case)

    def test_raw_time_order_and_spatial_screen_retain_failure(self):
        cfg=self.protocol();times=[np.arange(n+1)*(.0005/n) for n in cfg['time_steps']]
        reference=lambda t:np.column_stack([np.sin(2*np.pi*(i+1)*t/.0005) for i in range(3)])
        # Identical physical signal with an exact dt^2 perturbation at every level.
        traces=[reference(t)+(.02/(4**i))*np.cos(6*np.pi*t/.0005)[:,None] for i,t in enumerate(times)]
        rows,_=compare_raw(times,traces,cfg,'time')
        self.assertTrue(all(r['passed'] for r in rows))
        np.testing.assert_allclose([r['difference_ratio'] for r in rows],4.,atol=1e-10)
        t=times[1];base=reference(t);errors=np.cos(6*np.pi*t/.0005)[:,None]
        rows,_=compare_raw([t]*3,[base+.3*errors,base+.1*errors,base],cfg,'spatial')
        self.assertFalse(any(r['passed'] for r in rows))
        rows,_=compare_raw([t]*3,[base+.1*errors,base+.02*errors,base],cfg,'spatial')
        self.assertTrue(all(r['passed'] for r in rows))
        rows,_=compare_raw([t]*3,[np.zeros_like(base)]*3,cfg,'spatial')
        self.assertFalse(any(r['rate_identifiable'] or r['passed'] for r in rows))
        bad=times.copy();bad[2]=bad[2].copy();bad[2][-1]-=1e-8
        with self.assertRaises(ValueError):compare_raw(bad,traces,cfg,'time')

    def test_independent_mass_and_state_energy_against_assembled_operator(self):
        plate=Plate(width_m=.04,height_m=.03,thickness_m=.003);material=Material()
        transform=np.array([[1,.13,.07],[.04,1,.11],[.02,.03,1.]])
        for p in (3,4,5):
            layout=make_layout(plate,(2,1,1),p,transform);grid=make_grid(plate,(2,1,1),p,transform)
            system=assemble_tensor(grid,material);n=layout.screen['vector_dofs']
            u,v=np.random.default_rng(p).normal(size=(2,n));u*=1e-9;v*=1e-3;dt=1e-9
            result=cpu_state(layout,material,u,v,dt)
            np.testing.assert_allclose(np.tile(result['scalar_mass_kg'],3),system.mass_kg,rtol=1e-12,atol=0)
            ku=system.stiffness@u;physical=.5*(v@(system.mass_kg*v)+u@ku)
            modified=physical-dt**2/8*(ku@(ku/system.mass_kg))
            self.assertLess(abs(result['physical_energy_j']/physical-1),1e-12)
            self.assertLess(abs(result['modified_energy_j']/modified-1),1e-12)

    def test_archive_header_rejects_large_claim_before_array_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'arrays.npz'
            np.savez_compressed(path,values=np.arange(3,dtype=float))
            archive_schema(path,sha256_file(path),{'values':(3,)})
            with self.assertRaises(ValueError):archive_schema(path,'0'*64,{'values':(3,)})
            header=BytesIO();np.lib.format.write_array_header_1_0(header,dict(descr='<f8',fortran_order=False,shape=(10**12,)))
            with zipfile.ZipFile(path,'w') as archive:archive.writestr('values.npy',header.getvalue()+b'0'*24)
            with self.assertRaises(ValueError):archive_schema(path,sha256_file(path),{'values':(3,)})
            with zipfile.ZipFile(path,'w') as archive:archive.writestr('unexpected.npy',b'0')
            with self.assertRaises(ValueError):archive_schema(path,sha256_file(path),{'values':(3,)})


if __name__=='__main__':unittest.main()
