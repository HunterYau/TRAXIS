"""Build the same spectral mesh with precomputed dense GPU patch actions."""
import numpy as np
from traxis.config import Material
from traxis.gpu_tensor import build_gpu_tensor
from traxis.gpu_enriched_verlet import sparse_probe_set
from traxis.pzt_receiver import DiskGeometry
from traxis.piezoelectric import PiezoMaterial, read_profile
from traxis.spectral_pzt_sparse import component
from traxis.quadratic_pzt_sparse import SparseAttachment, ConsistentMass
from traxis.gpu_pzt_precomputed import couple_precomputed


def build(plan, row, guard, event):
    import cupy as cp
    from examples import pzt_mesh_protocol as protocol
    from examples.pzt_mesh_study import available_host_bytes
    if row.get('element') != 'spectral' or not plan['execution']['skip_stability_checks']:
        raise ValueError('Precomputed adapter currently supports the existing unchecked spectral study')
    c,e = plan['configuration'],row['estimate']
    if c['body']['bond']!='perfect': raise ValueError('Existing perfect interface required')
    if e['estimated_host_peak_bytes'] > .8*available_host_bytes():
        raise MemoryError('Precomputed study needs more host RAM; use sparse fallback or larger cloud host')
    if e['estimated_gpu_peak_bytes'] > min(.8*cp.cuda.runtime.memGetInfo()[0],plan['protocol']['device_budget_mib']*2**20):
        raise MemoryError('Precomputed study needs more GPU memory; use sparse fallback or larger cloud GPU')
    layout=protocol.layout_for(c)
    layout.screen['max_device_mib']=plan['protocol']['device_budget_mib']
    base=build_gpu_tensor(layout,Material(**c['host_material']),guard=guard)
    material=PiezoMaterial.from_profile(read_profile(protocol.PROJECT/c['material_profile']))
    def geo(center,surface):
        return DiskGeometry(tuple(center),c['disk_radius_m'],c['disk_thickness_m'],surface,1)
    event('assembling_shared_spectral_pzt')
    data=component(geo(c['actuator_center_xy_m'],'top'),material,
        rings=row['radial'],sectors=row['angular'],layers=row['thickness'],
        max_workspace_mib=1536,max_factor_mib=1536)
    bodies=[];masses=[];next_dof=layout.screen['vector_dofs']
    for center,surface,electrical in [(c['actuator_center_xy_m'],'top','voltage'),
            (c['receiver_center_xy_m'],'top','open'),(c['receiver_center_xy_m'],'bottom','open')]:
        event('factoring_sparse_consistent_mass',body=len(bodies)+1)
        b=SparseAttachment(data,layout,geo(center,surface),first_dof=next_dof,electrical=electrical)
        masses.append(ConsistentMass(b,base.mass_diagonal,max_factor_mib=1536))
        bodies.append(b);next_dof=b.next_dof
        if guard: guard.check(force=True)
    system=couple_precomputed(base,bodies,masses,guard=guard,progress=event,
        max_workspace_mib=plan['protocol']['component_workspace_mib'],block_columns=e['block_columns'])
    if len(system.mass_diagonal)!=e['total_mechanical_dofs'] or [len(b.indices) for b in bodies]!=e['patch_dofs']:
        raise ArithmeticError('Precomputed DOF/support forecast mismatch')
    force=np.zeros(next_dof);force[bodies[0].indices]=-bodies[0].coupling
    probes=sparse_probe_set([(b.indices,b.coupling/b.total_capacitance_f) for b in bodies[1:]],next_dof)
    return system,force,probes
