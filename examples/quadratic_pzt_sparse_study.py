"""Construction/forecast hooks only; the original LTS recurrence is untouched."""
import numpy as np
from traxis.config import Material
from traxis.gpu_tensor import build_gpu_tensor
from traxis.gpu_enriched_verlet import sparse_probe_set
from traxis.pzt_receiver import DiskGeometry
from traxis.piezoelectric import PiezoMaterial,read_profile
from traxis.quadratic_pzt_sparse import component,SparseAttachment,ConsistentMass,receipt


def estimate(layout,e):
    """Conservative admission forecast using explicit per-factor budgets.

    Actual fill is measured at construction; SuperLU workspace is NOT bounded
    rigorously. This upper payload forecast duplicates shared components.
    """
    e=dict(e);n=e['total_mechanical_dofs'];ne=e['body_tetrahedra_each']
    payload=6*(ne*1400*12*2+2*1536*2**20)
    return dict(e,dense_reference_gpu_peak_bytes=e['estimated_gpu_peak_bytes'],
        dense_reference_host_peak_bytes=e['estimated_host_peak_bytes'],
        estimated_gpu_peak_bytes=int(layout.screen['device_storage_screen_mib']*2**20+payload+40*n*8),
        estimated_host_peak_bytes=int(payload+6*n*8+1536*2**20),
        attachment_workspace_bytes=1536*2**20,merge_workspace_bytes=0,
        estimate_note='Sparse topology triplet upper count plus explicit LU factor budgets and duplicated shared-component margin; not a measured peak. No dense inverse or electrical Schur matrix.',
        estimated_case_total_seconds_range=[None,None],
        timing_note='No full sparse run benchmark; factor setup and sparse solves have different cost. Dense cubic scaling is inapplicable.')


def build(plan,row,guard,event):
    import cupy as cp
    from examples import pzt_mesh_protocol as protocol
    from examples.pzt_mesh_study import available_host_bytes
    from traxis.gpu_quadratic_pzt_sparse import couple_sparse
    if row.get('element')!='quadratic':raise ValueError('Sparse study requires quadratic elements')
    execution=plan['execution']
    if not execution['skip_stability_checks'] or execution['backend']!='original':
        raise ValueError('Sparse adapter requires original unchecked runner; dense graphs/screening adapters are unsupported')
    c=plan['configuration'];e=row['estimate']
    if c['body']['bond']!='perfect':raise ValueError('Only unchanged perfect interface is supported')
    if e['estimated_host_peak_bytes']>.8*available_host_bytes():raise MemoryError('Sparse host-memory forecast exceeds allocation')
    if e['estimated_gpu_peak_bytes']>min(.8*cp.cuda.runtime.memGetInfo()[0],plan['protocol']['device_budget_mib']*2**20):
        raise MemoryError('Sparse device-memory forecast exceeds allocation')
    layout=protocol.layout_for(c);layout.screen['max_device_mib']=plan['protocol']['device_budget_mib']
    base=build_gpu_tensor(layout,Material(**c['host_material']),guard=guard)
    material=PiezoMaterial.from_profile(read_profile(protocol.PROJECT/c['material_profile']))
    def geo(center,surface):return DiskGeometry(tuple(center),c['disk_radius_m'],c['disk_thickness_m'],surface,1)
    event('assembling_shared_sparse_pzt')
    data=component(geo(c['actuator_center_xy_m'],'top'),material,
        rings=row['radial'],sectors=row['angular'],layers=row['thickness'],
        max_workspace_mib=1536,max_factor_mib=1536)
    guard.check(force=True);bodies=[];masses=[];next_dof=layout.screen['vector_dofs']
    for center,surface,electrical in [(c['actuator_center_xy_m'],'top','voltage'),
        (c['receiver_center_xy_m'],'top','open'),(c['receiver_center_xy_m'],'bottom','open')]:
        event('factoring_sparse_consistent_mass',body=len(bodies)+1)
        b=SparseAttachment(data,layout,geo(center,surface),first_dof=next_dof,electrical=electrical)
        m=ConsistentMass(b,base.mass_diagonal,max_factor_mib=1536)
        event('sparse_component_saved',body=len(bodies)+1,storage=receipt(b,m))
        bodies.append(b);masses.append(m);next_dof=b.next_dof;guard.check(force=True)
    system=couple_sparse(base,bodies,masses,guard=guard,factor_backend=execution.get('pzt_factor_backend','gpu'))
    if len(system.mass_diagonal)!=e['total_mechanical_dofs']:raise ArithmeticError('Sparse DOF forecast mismatch')
    if [len(b.indices) for b in bodies]!=e['patch_dofs']:raise ArithmeticError('Sparse support forecast mismatch')
    force=np.zeros(next_dof);force[bodies[0].indices]=-bodies[0].coupling
    probes=sparse_probe_set([(b.indices,b.coupling/b.total_capacitance_f) for b in bodies[1:]],next_dof)
    return system,force,probes
