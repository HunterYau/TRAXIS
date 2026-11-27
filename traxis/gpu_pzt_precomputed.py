"""Upload precomputed PZT matrices to UNCHANGED dense/graph LTS execution."""
import math
from types import SimpleNamespace
import numpy as np
from traxis.gpu_elements import GPUElementOperator
from traxis.gpu_pzt_body import BodyStorage
from traxis.pzt_precomputed import matrices


def couple_precomputed(base, attachments, masses, *, guard=None, progress=None,
                       max_workspace_mib=16384, block_columns=64):
    import cupy as cp
    if not attachments or len(attachments) != len(masses):
        raise ValueError('Matching attachments and masses required')
    next_dof = len(base.mass_diagonal)
    seen = set()
    for a in attachments:
        if seen.intersection(a.indices.tolist()):
            raise ValueError('Precomputed adapter requires disjoint mass supports')
        seen.update(a.indices.tolist())
        if not np.array_equal(a.indices[a.indices>=len(base.mass_diagonal)],np.arange(next_dof,a.next_dof)):
            raise ValueError('Nonconsecutive PZT DOFs')
        next_dof = a.next_dof
    diagonal = np.r_[base.mass_diagonal,np.zeros(next_dof-len(base.mass_diagonal))]
    for a,m in zip(attachments,masses): diagonal[a.indices] = m.diagonal
    extra = sum(3*len(a.indices)**2*8+a.indices.nbytes for a in attachments)+3*diagonal.nbytes
    device_bytes = base.screen['device_storage_screen_mib']*2**20+extra
    if (device_bytes>base.screen['max_device_mib']*2**20
            or extra>.8*cp.cuda.runtime.memGetInfo()[0]):
        raise MemoryError('Precomputed PZT matrices exceed GPU allocation; use sparse fallback or a larger cloud GPU')
    stable = 2/math.sqrt(4/base.stable_dt_s**2+max(a.data.body_eigenvalue_bound for a in attachments))
    screen = dict(base.screen, vector_dofs=len(diagonal), stable_dt_s=stable,
        device_storage_screen_mib=device_bytes/2**20, body_device_storage_mib=extra/2**20,
        coupled_mass='exact full consistent mass, precomputed inverse from retained sparse LU',
        pzt_storage='precomputed', body_reports=[a.report for a in attachments],
        precomputed_verification=[], mass_inverse_errors=[])
    records = []
    with base.stiffness.storage.lock,cp.cuda.Device(base.stiffness.storage.device_id):
        # Stream one host patch at a time. Keep the original dense storage and
        # CUDA kernel, including its reduction order, entirely unchanged.
        storage = BodyStorage(base.stiffness.storage,diagonal,[],screen,guard)
        for i,(a,m) in enumerate(zip(attachments,masses),1):
            if guard: guard.check(force=True)
            def event(phase, **info):
                if progress: progress(phase,body=i,**info)
            p = matrices(a,m,block_columns=block_columns,max_workspace_mib=max_workspace_mib,progress=event)
            storage.patches.append((cp.asarray(p.indices),cp.asarray(p.stiffness,order='C'),
                cp.asarray(p.mass,order='C'),cp.asarray(p.inverse,order='C')))
            cp.cuda.get_current_stream().synchronize()
            screen['precomputed_verification'].append(p.verification)
            screen['mass_inverse_errors'].append(p.inverse_error)
            records.append(SimpleNamespace(indices=a.indices))
            del p
            event('precomputed_patch_uploaded')
    return SimpleNamespace(stiffness=GPUElementOperator(storage,1.,0.),mass_diagonal=diagonal,
        stable_dt_s=stable,screen=screen,layout=base.layout,device=base.device,
        body_attachments=attachments,body_patches=records,
        inverse_mass_inplace=storage.inverse_mass_inplace,mass_quadratic=storage.mass_quadratic)
