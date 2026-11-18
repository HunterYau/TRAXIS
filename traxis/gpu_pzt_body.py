"""Device-resident resolved PZT attachments for the tensor plate solver."""
import math
from types import SimpleNamespace

import numpy as np

from traxis.gpu_elements import _Storage, GPUElementOperator
from traxis.pzt_body import merge_attachments


DENSE_KERNEL = r'''
extern "C" __global__ void local_matvec(const double* a, const double* x,
                                      double* y, int n) {
    int row=blockIdx.x, lane=threadIdx.x;
    __shared__ double sums[128];
    double sum=0.;
    for(int col=lane;col<n;col+=128) sum+=a[(long long)row*n+col]*x[col];
    sums[lane]=sum; __syncthreads();
    for(int stride=64;stride>0;stride/=2) {
        if(lane<stride) sums[lane]+=sums[lane+stride];
        __syncthreads();
    }
    if(lane==0) y[row]=sums[0];
}
'''


class BodyStorage(_Storage):
    def __init__(self, base, diagonal, patches, screen, guard):
        self.base = base
        self.cp, self.device_id, self.lock = base.cp, base.device_id, base.lock
        self.guard, self.screen = guard, screen
        self.ns = len(diagonal)//3; self.host_dofs = 3*base.ns
        self.applications = self.vector_uploads = self.vector_downloads = 0
        self.preconditioner_applications = 0
        cp = self.cp
        self.input = cp.empty(len(diagonal), dtype=cp.float64)
        self.output = cp.empty_like(self.input)
        self.diagonal = cp.asarray(diagonal)
        self.patches = [(cp.asarray(p.indices), cp.asarray(p.stiffness,order='C'),
            cp.asarray(p.mass,order='C'), cp.asarray(p.inverse,order='C')) for p in patches]
        self.dense_kernel=cp.RawKernel(DENSE_KERNEL,'local_matvec',options=('--std=c++11',))

    def matvec(self, matrix, values):
        answer=self.cp.empty_like(values); n=len(values)
        self.dense_kernel((n,),(128,),(matrix,values,answer,np.int32(n)))
        return answer

    def _launch(self, values, result, stiffness_scale=0., mass_scale=0., *, inverse=False):
        if inverse or mass_scale != 0:
            raise ValueError('use the coupled mass operations explicitly')
        result.fill(0.)
        self.base._launch(values[:self.host_dofs], result[:self.host_dofs], stiffness_scale, 0.)
        for ids, stiffness, _, _ in self.patches:
            result[ids] += stiffness_scale*self.matvec(stiffness,values[ids])
        self.applications += 1

    def inverse_mass_inplace(self, values):
        local = [values[ids] for ids, _, _, _ in self.patches]
        values /= self.diagonal
        for (ids, _, _, inverse), original in zip(self.patches, local):
            values[ids] = self.matvec(inverse,original)

    def mass_quadratic(self, values, *, inverse=False):
        cp = self.cp
        diagonal = 1/self.diagonal if inverse else self.diagonal
        answer = cp.sum(diagonal*values*values, dtype=cp.float64)
        for ids, _, mass, inv in self.patches:
            local = values[ids]
            answer += cp.sum(local*self.matvec(inv if inverse else mass,local))-cp.sum(diagonal[ids]*local*local)
        return answer


def couple_gpu_bodies(base, attachments, *, max_workspace_mib=512, guard=None):
    """Keep the host GLL mass and exact consistent body mass in separate blocks."""
    diagonal, patches = merge_attachments(base.mass_diagonal, attachments,
        max_workspace_mib=max_workspace_mib, guard=guard)
    # Plate K bound survives because the added body mass is positive semidefinite.
    # Disjoint patch terms are bounded by their largest generalized eigenvalue.
    bound = 4/base.stable_dt_s**2 + max(p.eigenvalue_upper_s2 for p in patches)
    stable = 2/math.sqrt(bound)
    extra_bytes = sum(3*p.mass.nbytes+p.indices.nbytes for p in patches)+24*len(diagonal)
    budget = base.screen['device_storage_screen_mib']+extra_bytes/1024**2
    if budget > base.screen['max_device_mib']:
        raise MemoryError('PZT body storage exceeds declared total device budget')
    cp = base.stiffness.storage.cp
    screen = {**base.screen, 'vector_dofs':len(diagonal), 'stable_dt_s':stable,
        'device_storage_screen_mib':budget, 'body_device_storage_mib':extra_bytes/1024**2,
        'coupled_mass':'GLL host plus exact transformed consistent PZT mass',
        'body_reports':[b.report for b in attachments],
        'mass_inverse_errors':[p.inverse_error for p in patches]}
    with base.stiffness.storage.lock, cp.cuda.Device(base.stiffness.storage.device_id):
        if guard: guard.check(force=True)
        free, _ = cp.cuda.runtime.memGetInfo()
        if extra_bytes > .8*free:
            raise MemoryError('PZT body storage exceeds available GPU memory screen')
        storage = BodyStorage(base.stiffness.storage, diagonal, patches, screen, guard)
    return SimpleNamespace(stiffness=GPUElementOperator(storage, 1., 0.),
        mass_diagonal=diagonal, stable_dt_s=stable, screen=screen, layout=base.layout,
        device=base.device, body_attachments=attachments, body_patches=patches,
        inverse_mass_inplace=storage.inverse_mass_inplace,
        mass_quadratic=storage.mass_quadratic)
