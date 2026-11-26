"""Q2 spectral component adapter for the existing sparse coupled LTS system."""
import math
import time
from types import SimpleNamespace
import numpy as np
import psutil
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import splu
from .spectral_piezoelectric import disk_mesh,assemble,element_matrices
from .quadratic_pzt_sparse import sparse_bytes,factor_bytes


def component(geometry,material,*,rings=6,sectors=64,layers=1,
              max_workspace_mib=768,max_factor_mib=512):
    if sectors%4:raise ValueError('Spectral polygon sides must be divisible by four')
    tick=time.perf_counter()
    mesh=disk_mesh(geometry,side_cells=sectors//4,radial_cells=rings,layers=layers,order=2)
    a=assemble(mesh,material,quadrature_order=6,max_workspace_mib=max_workspace_mib)
    n=len(mesh.points_m);z=mesh.points_m[:,2]
    lower=np.flatnonzero(abs(z-z.min())<1e-14);upper=np.flatnonzero(abs(z-z.max())<1e-14)
    inside=np.setdiff1d(np.arange(n),np.r_[lower,upper]);di=a.dielectric[inside][:,inside].tocsc()
    if 64*sparse_bytes(di)>min(max_factor_mib*2**20,.35*psutil.virtual_memory().available):
        raise MemoryError('Electrical LU fill forecast exceeds component budget')
    lu=splu(di,permc_spec='MMD_AT_PLUS_A')
    if factor_bytes(lu)>max_factor_mib*2**20:raise MemoryError('Electrical factors exceed budget')
    psi=np.zeros(n);psi[upper]=1.;psi[inside]=-lu.solve(a.dielectric[inside]@psi)
    cap=float(psi@a.dielectric@psi)
    if not math.isfinite(cap) or cap<=0:raise ArithmeticError('Invalid electrode capacitance')
    c,e,eps=material.stress_charge
    effective=SimpleNamespace(stress_charge=(c+e.T@np.linalg.solve(eps,e),e,eps),density_kg_m3=material.density_kg_m3)
    bound=0.
    for hexahedron in mesh.hexahedra:
        k,m,_,_,_=element_matrices(mesh.points_m[hexahedron],effective,2,6)
        mass=sparse.block_diag([m,m,m]).toarray()
        bound=max(bound,float(eigh(k,mass,eigvals_only=True,subset_by_index=[80,80])[0]))
    return SimpleNamespace(mesh=mesh,n=n,lower=lower,upper=upper,inside=inside,
        stiffness=a.stiffness.tocsr(),scalar_mass=a.mass[:n,:n].tocsr(),
        bi=a.coupling[:,inside].tocsr(),di=di,electrical_lu=lu,g=a.coupling@psi,
        capacitance_f=cap,volume_m3=a.volume_m3,body_eigenvalue_bound=bound*(1+1e-8),
        seconds=time.perf_counter()-tick,
        report=dict(element='Q2 spectral hexahedron',nodes=n,hexahedra=len(mesh.hexahedra),
            rings=rings,sectors=sectors,layers=layers,quadrature_order=6,mass='full consistent',
            material_id=material.id,radius_m=geometry.radius_m,thickness_m=geometry.thickness_m))
