"""Preparation-only PZT refinement protocol and conservative resource forecasts."""
import ast
import copy
import json
import math
from pathlib import Path
import shutil
import numpy as np
from examples import pzt_local_time_study as lts
from examples.pzt_broadband import reconstruct
from traxis.config import Plate
from traxis.gpu_tensor import make_layout
from traxis.pzt_body import disk_mesh
from traxis.quadratic_piezoelectric import quadratic_disk
from traxis.pzt_receiver import DiskGeometry,_point_weights
from traxis.storage import sha256_file,write_json,canonical_hash

PROJECT=Path(__file__).resolve().parents[1]
CONFIG=PROJECT/'configs/pzt-mesh-convergence'
BASE=dict(radial=5,angular=48,thickness=2)
LEGACY_BASE=dict(radial=6,angular=32,thickness=3)
REQUESTED_COARSE_DT_S=.00035/32768
LEVELS=[dict(BASE),dict(radial=6,angular=64,thickness=2),dict(radial=8,angular=64,thickness=2)]


def cases(element='quadratic'):
    if element=='spectral':
        return [dict(order=i,id=f'{i:02d}-spectral-q2-r{r}',family='combined',level=i,
                     element='spectral',radial=r,angular=64,thickness=1) for i,r in enumerate((6,8),1)]
    if element!='quadratic':raise ValueError('Unsupported PZT element')
    return [dict(order=level,id=f"{level:02d}-quadratic-{('baseline','medium','fine')[level-1]}",family='combined',level=level,element='quadratic',**values)
            for level,values in enumerate(LEVELS,1)]


def fingerprints():
    paths=[*sorted((PROJECT/'traxis').glob('*.py')),*sorted((PROJECT/'examples').glob('*.py')),
           *sorted(CONFIG.glob('*.json')),*sorted((PROJECT/'configs/pzt-mesh-spectral').glob('*.json')),PROJECT/'configs/pzt/nce51-noliac-ver1404.json']
    return {p.relative_to(PROJECT).as_posix():sha256_file(p) for p in paths}


def mathematical_ast(source):
    """Ignore only resource guards/options to verify unchanged assembly formulas."""
    class ResourceOnly(ast.NodeTransformer):
        def visit_If(self,node):
            if len(node.body)==1 and isinstance(node.body[0],ast.Raise):return None
            return self.generic_visit(node)
        def visit_arguments(self,node):
            pairs=[(a,d) for a,d in zip(node.kwonlyargs,node.kw_defaults) if not a.arg.startswith('max_')]
            node.kwonlyargs=[a for a,_ in pairs];node.kw_defaults=[d for _,d in pairs]
            return self.generic_visit(node)
        def visit_Call(self,node):
            node.keywords=[k for k in node.keywords if not (k.arg or '').startswith('max_')]
            return self.generic_visit(node)
    tree=ResourceOnly().visit(ast.parse(source))
    return canonical_hash(ast.dump(tree,include_attributes=False))


def verify_baseline(folder,assembly_sources):
    folder=Path(folder);p=json.loads((folder/'plan.json').read_text())
    if (json.loads((folder/'status.json').read_text())['status']!='completed'
        or p['case']['body']!=[32,6,3] or p['substeps']!=2 or p['source_sampling']!='coarse'
        or p['frequencies_hz']!=lts.FREQUENCIES or p['plate_condition']!='intact'
        or p['coarse_steps']!=32768 or p['coarse_dt_s']!=.00035/32768
        or p['case']['cells']!=250 or p['case'].get('nz',1)!=1 or p['case']['steps']!=65536):
        raise ValueError('Baseline is not the completed six-frequency 6/32/3 LTS study')
    if not json.loads((folder/'lts-kernel/result.json').read_text())['completed']:raise ValueError('Incomplete baseline kernel')
    trace=folder/'lts-kernel/trace.npz'
    if sha256_file(trace)!=json.loads((folder/'lts-kernel/result.json').read_text())['sha256']:
        raise ValueError('Baseline kernel checksum mismatch')
    compatibility={}
    with np.load(trace,allow_pickle=False) as a:kernel=a['voltage_v'].copy()
    for name,expected in p['identity']['sources'].items():
        if name=='examples/pzt_local_time_study.py':continue # Same explicit excitation verified below.
        path=PROJECT/name
        if name in ('traxis/pzt_body.py','traxis/piezoelectric.py'):
            old=Path(assembly_sources)/Path(name).name
            if sha256_file(old)!=expected or mathematical_ast(old.read_text())!=mathematical_ast(path.read_text()):
                raise ValueError('PZT assembly formulas differ from completed baseline: '+name)
            compatibility[name]=dict(baseline_sha256=expected,current_sha256=sha256_file(path),
                mathematical_ast_sha256=mathematical_ast(path.read_text()),resource_guards_only=True)
        elif not path.exists() or sha256_file(path)!=expected:
            raise ValueError('Physical/integration source differs from baseline: '+name)
    for f in lts.FREQUENCIES:
        row=folder/'lts-broadband'/f'{f//1000}khz'
        if json.loads((row/'source.json').read_text())!=lts.excitation(f,p['case']['steps'])[1]:
            raise ValueError('Baseline excitation differs')
        with np.load(row/'waveforms.npz',allow_pickle=False) as a:
            t,y=a['time_s'],a['voltage_v']
            if (t.shape!=(32769,) or y.shape!=(32769,2) or not np.isfinite(y).all()
                    or not np.allclose(t,np.arange(32769)*p['coarse_dt_s'],rtol=0,atol=1e-15)):
                raise ValueError('Baseline waveform grid or values invalid')
            expected=reconstruct(kernel,lts.excitation(f,p['case']['steps'])[0](t))
            relative=float(np.linalg.norm(y-expected)/np.linalg.norm(expected))
            if not math.isfinite(relative) or relative>1e-12:
                raise ValueError('Saved baseline waveform does not match its kernel and declared excitation')
            compatibility[f'baseline/{f//1000}khz']=dict(reconstruction_relative_l2=relative)
    return p,compatibility


def layout_for(config):
    return make_layout(Plate(*config['plate_m']),tuple(config['mesh']['cells']),config['mesh']['order'],
                       max_dofs=32000000,max_device_mib=6144,max_host_mib=4096)


def geometric_estimate(config,row):
    """Coordinates/topology/support only; no dense PZT assembly and no GPU."""
    layout=layout_for(config);s,r,z=row['angular'],row['radial'],row['thickness']
    geometry=DiskGeometry(tuple(config['actuator_center_xy_m']),config['disk_radius_m'],config['disk_thickness_m'],'top',1)
    if row.get('element')=='spectral':
        from traxis.spectral_piezoelectric import disk_mesh as hex_mesh,basis
        mesh=hex_mesh(geometry,side_cells=s//4,radial_cells=r,layers=z,order=2)
        points=mesh.points_m;tets=mesh.hexahedra
        ref=basis(2,6)[1]
        jac=(ref[None]@points[tets][:,None]).reshape(-1,3,3)
    elif row.get('element')=='quadratic':
        mesh=quadratic_disk(geometry,sectors=s,rings=r,layers=z,max_nodes=20000)
    else:
        mesh,_=disk_mesh(geometry,sectors=s,rings=r,layers=z,max_nodes=8192,max_elements=65536)
    if row.get('element')!='spectral':
        points=mesh.points_m;tets=mesh.tetrahedra
        corners=tets[:,:4]
        jac=(points[corners[:,1:]]-points[corners[:,0,None]]).transpose(0,2,1)
    contact_nodes=np.flatnonzero(abs(points[:,2]-points[:,2].min())<1e-14)
    free_dofs=3*(len(points)-len(contact_nodes))
    singular=np.linalg.svd(jac,compute_uv=False)
    if np.any(singular[:,-1]<=0) or np.max(singular[:,0]/singular[:,-1])>1e8:raise ValueError('Invalid mesh quality')
    patches=[]
    for center,surface in ((config['actuator_center_xy_m'],'top'),(config['receiver_center_xy_m'],'top'),
                           (config['receiver_center_xy_m'],'bottom')):
        contact=points[contact_nodes,:2]+np.array(center);host=set()
        for xy in contact:
            ids,_=_point_weights(layout,(*xy,config['plate_m'][2] if surface=='top' else 0.))
            nz=layout.cells[2]*layout.order+1
            ids=ids[ids%nz==(nz-1 if surface=='top' else 0)]
            host.update(ids.tolist())
        patches.append(3*len(host)+free_dofs)
    ndof=layout.screen['vector_dofs']+3*free_dofs
    squares=sum(n*n for n in patches);retained=2*squares*8
    attachment=max((12*(3*len(points))**2+10*n*n)*8 for n in patches)
    merge=10*squares*8
    host=1.25*(max(attachment+retained,merge+retained)+48*ndof*8)
    storage=layout.screen['device_storage_screen_mib']*1024**2+3*squares*8+24*ndof
    gpu=storage+max(48*ndof*8+3*squares*8,32*ndof*8)
    return dict(body_nodes_each=len(points),body_tetrahedra_each=0 if row.get('element')=='spectral' else len(tets),body_elements_each=len(tets),
        body_hexahedra_each=len(tets) if row.get('element')=='spectral' else 0,
        element=row.get('element','linear'),body_contact_nodes_each=len(contact_nodes),added_mechanical_dofs=3*free_dofs,total_mechanical_dofs=ndof,patch_dofs=patches,
        min_tetra_singular_m=float(singular[:,-1].min()),max_tetra_condition=float((singular[:,0]/singular[:,-1]).max()),
        attachment_workspace_bytes=attachment,merge_workspace_bytes=merge,
        estimated_host_peak_bytes=int(host),estimated_gpu_peak_bytes=int(1.15*gpu),
        estimate_note='Dense consistent PZT matrices retained; forecasts, not measured peaks; runtime rechecks actual available memory.')


def prepare(root,baseline,assembly_sources,*,start_level=1,primary_only=False,coarse_dt_s=None,skip_stability_checks=False,execution_backend='optimized',pzt_storage='dense',pzt_factor_backend='gpu',element='quadratic'):
    config_folder=CONFIG if element=='quadratic' else PROJECT/'configs/pzt-mesh-spectral'
    if element=='spectral' and pzt_storage not in ('sparse','precomputed'):raise ValueError('Spectral study requires sparse or precomputed PZT storage')
    if pzt_storage=='precomputed' and (element!='spectral' or not skip_stability_checks):
        raise ValueError('Precomputed storage requires the existing unchecked spectral study')
    if execution_backend not in ('original','optimized','optimized-no-graphs'):raise ValueError('Invalid execution backend')
    if pzt_storage not in ('dense','sparse','precomputed'):raise ValueError('Invalid PZT storage')
    if pzt_factor_backend not in ('gpu','cpu'):raise ValueError('Invalid PZT factor backend')
    if pzt_storage=='sparse' and (not skip_stability_checks or execution_backend!='original'):
        raise ValueError('Sparse study requires --skip-stability-checks and --execution-backend original; LTS unchanged')
    if skip_stability_checks:
        start_level=1;primary_only=True;coarse_dt_s=REQUESTED_COARSE_DT_S
    root=Path(root)
    if start_level not in range(1,len(LEVELS)+1):raise ValueError('Invalid starting combined mesh level')
    if coarse_dt_s is not None and (not math.isfinite(coarse_dt_s) or coarse_dt_s<=0):raise ValueError('Positive finite fixed step required')
    if root.exists():raise FileExistsError('Use a new study output directory')
    old,compatibility=verify_baseline(baseline,assembly_sources)
    protocol=json.loads((config_folder/'protocol.json').read_text())
    rows=cases(element)
    for row in rows:
        if json.loads((config_folder/(row['id']+'.json')).read_text())!=row:raise ValueError('Configuration table changed')
    c=copy.deepcopy(old['configuration'])
    c['mesh']['cells']=[old['case']['cells'],old['case']['cells'],old['case'].get('nz',1)]
    if c.get('setup_profile') or c.get('damping_profile') or c['body']['bond']!='perfect':
        raise ValueError('Unexpected physical setup or damping profile')
    base=geometric_estimate(c,dict(LEGACY_BASE,id='historical-linear-template'))
    stable=None
    console=Path(baseline)/'console.log'
    for line in (console.read_text().splitlines() if console.exists() else []):
        try:
            data=json.loads(line)
            if data.get('phase')=='bodies_ready':stable=data['stable_dt_s']
        except ValueError:pass
    if stable is None and not skip_stability_checks:raise ValueError('Missing recorded baseline stability limit')
    build_seconds=json.loads((Path(baseline)/'build.json').read_text())['seconds']
    solve_seconds=json.loads((Path(baseline)/'lts-kernel/result.json').read_text())['seconds']
    for row in rows:
        e=geometric_estimate(c,row);ratio=min(1.,e['min_tetra_singular_m']/base['min_tetra_singular_m'])
        H=coarse_dt_s if skip_stability_checks else old['coarse_dt_s']
        k=round(math.log2(old['coarse_dt_s']/H))
        while not skip_stability_checks and H/2>.8*stable*ratio:H/=2;k+=1
        matrix_ratio=sum(v*v for v in e['patch_dofs'])/sum(v*v for v in base['patch_dofs'])
        dense_ratio=(e['body_nodes_each']/base['body_nodes_each'])**3
        low=solve_seconds*2**k*(.95+.05*matrix_ratio)
        high=solve_seconds*2**k*(.5+.5*matrix_ratio)
        e.update(provisional_coarse_dt_s=H,provisional_fine_dt_s=H/2,
            provisional_primary_steps=old['coarse_steps']*2**k,
            provisional_half_step_steps=old['coarse_steps']*2**(k+1),
            estimated_primary_kernel_seconds_range=[low,high],
            estimated_case_total_seconds_range=[3*low+.5*build_seconds*dense_ratio,3*high+3*build_seconds*dense_ratio],
            timing_note='Projection from recorded 237 s LTS solve and 10 s build, with unmeasured 5-50% dense-patch work fraction and cubic assembly scaling. Excludes unknown screen and IO overhead; actual stability may require more steps.')
        if skip_stability_checks:
            e.update(provisional_half_step_steps=0,
                estimated_case_total_seconds_range=[low+.5*build_seconds*dense_ratio,high+3*build_seconds*dense_ratio],
                timing_note='Fixed requested steps, one kernel only; forecast excludes plotting/IO and is not measured or stability certified.')
        if pzt_storage=='sparse':
            from examples.quadratic_pzt_sparse_study import estimate
            e=estimate(layout_for(c),e)
        elif pzt_storage=='precomputed':
            from traxis.pzt_precomputed import estimate
            e=estimate(layout_for(c),e)
        row['estimate']=e
    root.mkdir(parents=True)
    selected=['plan.json','status.json','partition.json','performance.json','build.json','stability-screen.json',
              'lts-kernel/trace.npz','lts-kernel/result.json']
    selected += [f'lts-broadband/{f//1000}khz/{name}' for f in lts.FREQUENCIES for name in ('waveforms.npz','source.json')]
    for name in selected:
        target=root/'baseline'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(baseline)/name,target)
    for name in ('pzt_body.py','piezoelectric.py'):
        target=root/'baseline-assembly-sources'/name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(Path(assembly_sources)/name,target)
    hashes={p.relative_to(root/'baseline').as_posix():sha256_file(p) for p in (root/'baseline').rglob('*') if p.is_file()}
    plan=dict(schema=1,element=element,scope='PZT mesh convergence, intact plate, unchanged LTS split-LFC broadband integration',
        protocol=protocol,cases=rows,configuration=c,baseline_hashes=hashes,baseline_compatibility=compatibility,
        baseline_coarse_dt_s=old['coarse_dt_s'],baseline_coarse_steps=old['coarse_steps'],duration_s=.00035,
        edge_width_m=old['edge_width_m'],boxes=old['boxes'],source_sampling='coarse',substeps=2,
        identity=fingerprints(),baseline_rerun=False,simulation_started=False,comparison_baseline_case=rows[0]['id'],quadratic_baseline_new=True,
        execution=dict(start_level=start_level,primary_only=primary_only,fixed_coarse_dt_s=coarse_dt_s,
                       skip_stability_checks=skip_stability_checks,backend=execution_backend,pzt_storage=pzt_storage,pzt_factor_backend=pzt_factor_backend),
        temporal_comparison='Not performed; primary-only execution.' if primary_only else 'Each mesh primary vs half step; target 0.01% for raw/envelope/wiggle. Failed checks remain unresolved; no automatic extra meshes or optimal selection.',
        baseline_limit='Quadratic 5/48/2 is simulated first and used for all mesh comparisons. Archived linear 6/32/3 supplies fixed plate/source settings only; its waveforms are not the convergence reference. Temporal qualification is not performed in primary-only mode.')
    if element=='spectral':
        plan['quadratic_baseline_new']=False
        plan['spectral_baseline_new']=True
        plan['baseline_limit']='New Q2 6-ring plate baseline, then Q2 8-ring plate; archived tetrahedral data supplies fixed plate/source settings only. Fixed time steps are not temporally or stability certified.'
    write_json(root/'plan.json',plan);write_json(root/'status.json',dict(status='prepared',completed=[],failed=[]))
    lines=['Quadratic PZT convergence: run NEW baseline 5/48/2, then Medium 6/64/2, then Fine 8/64/2.',
        f'Execution: start at level {start_level}; primary only={primary_only}; fixed coarse dt={coarse_dt_s}. This execution policy overrides the default temporal protocol below.',
        'All three configurations use quadratic mechanical and electrical tetrahedra.',
        f'{len(LEVELS)+1-start_level} meshes, {(len(LEVELS)+1-start_level)*(1 if primary_only else 2)} impulse solves; six frequency reconstructions per solve.',
        f'Stability checks disabled={skip_stability_checks}; fixed time steps are not certified when disabled.',
        f'GPU execution backend: {execution_backend}; PZT storage: {pzt_storage}.',
        'Counts and geometry exact; memory/runtime are forecasts.',
        'Compare Medium and Fine to the new quadratic baseline, and Fine to Medium; archived linear results are not the convergence reference.',
        'Check the selected storage backend forecasts against declared budgets and available hardware.',
        'Order | Case | radial/angular/thickness | DOFs | GPU forecast GiB | RAM forecast GiB | provisional primary/half steps | case minutes range']
    for row in rows:
        if row['level']<start_level:continue
        e=row['estimate'];lo,hi=e['estimated_case_total_seconds_range']
        if primary_only and coarse_dt_s is not None:
            lines.append(f"{row['order']:02d} | {row['id']} | {row['radial']}/{row['angular']}/{row['thickness']} | "
                f"coarse={coarse_dt_s*1e9:.12g} ns fine={coarse_dt_s*1e9/2:.12g} ns | "
                f"{round(plan['duration_s']/coarse_dt_s)} steps; no half-step run | "
                f"DOFs={e['total_mechanical_dofs']} GPU forecast={e['estimated_gpu_peak_bytes']/1024**3:.1f} GiB")
            continue
        lines.append(f"{row['order']:02d} | {row['id']} | {row['radial']}/{row['angular']}/{row['thickness']} | {e['total_mechanical_dofs']} | {e['estimated_gpu_peak_bytes']/1024**3:.1f} | {e['estimated_host_peak_bytes']/1024**3:.1f} | {e['provisional_primary_steps']}/{e['provisional_half_step_steps']} | {lo/60:.1f}-{hi/60:.1f}")
    if element=='spectral':
        lines=[plan['baseline_limit'],'Q2 displacement and electrical potential; Gauss order 6; full consistent mass.',
            f'PZT storage={pzt_storage}; GPU execution={execution_backend}; same LTS integration; no extra half-step run.',
            '6-ring baseline then 8-ring refinement; six frequencies 50-300 kHz, four voltage channels.',
            'Meshes: annular rings / polygon sides / thickness cells = 6/64/1 and 8/64/1.',
            'Resource and stability estimates are not certification. Local full plate execution is not sized for 8 GiB.',*lines[10:]]
    (root/'schedule.txt').write_text('\n'.join(lines)+'\n')
    print((root/'schedule.txt').read_text(),flush=True)
    return plan


def validate(root,*,sources=True):
    root=Path(root);p=json.loads((root/'plan.json').read_text())
    if p['schema']!=1 or p['baseline_rerun'] is not False or p['substeps']!=2 or p['source_sampling']!='coarse':
        raise ValueError('Study protocol changed')
    element=p.get('element','quadratic')
    config_folder=CONFIG if element=='quadratic' else PROJECT/'configs/pzt-mesh-spectral'
    actual=[{k:v for k,v in row.items() if k!='estimate'} for row in p['cases']]
    if actual!=cases(element):raise ValueError('Scheduled PZT configurations changed')
    if p['protocol']!=json.loads((config_folder/'protocol.json').read_text()):raise ValueError('Protocol changed')
    if element=='spectral' and p.get('execution',{}).get('pzt_storage') not in ('sparse','precomputed'):raise ValueError('Spectral study requires sparse or precomputed storage')
    baseline_flag='spectral_baseline_new' if element=='spectral' else 'quadratic_baseline_new'
    if p.get('comparison_baseline_case')!=cases(element)[0]['id'] or p.get(baseline_flag) is not True:
        raise ValueError('New quadratic baseline must be simulated and used as reference')
    execution=p.get('execution',{})
    if execution.get('backend','original') not in ('original','optimized','optimized-no-graphs'):
        raise ValueError('Invalid GPU execution backend')
    if execution.get('start_level',1) not in range(1,len(LEVELS)+1) or type(execution.get('primary_only',False)) is not bool:raise ValueError('Invalid execution policy')
    if execution.get('pzt_storage','dense') not in ('dense','sparse','precomputed'):raise ValueError('Invalid PZT storage')
    if execution.get('pzt_storage')=='precomputed' and (element!='spectral' or not execution.get('skip_stability_checks')):
        raise ValueError('Precomputed storage requires the existing unchecked spectral study')
    if execution.get('pzt_factor_backend','gpu') not in ('gpu','cpu'):raise ValueError('Invalid PZT factor backend')
    if execution.get('pzt_storage')=='sparse' and (not execution.get('skip_stability_checks') or execution.get('backend')!='original'):
        raise ValueError('Sparse storage requires original unchecked LTS adapter')
    fixed=execution.get('fixed_coarse_dt_s')
    if type(execution.get('skip_stability_checks',False)) is not bool:raise ValueError('Invalid stability policy')
    if execution.get('skip_stability_checks') and (execution['start_level']!=1 or not execution['primary_only']
            or fixed!=REQUESTED_COARSE_DT_S):raise ValueError('Unchecked study requires the quadratic mesh and the requested original time pair')
    if fixed is not None and (not math.isfinite(fixed) or fixed<=0):raise ValueError('Invalid fixed step')
    if sources and p['identity']!=fingerprints():raise ValueError('Frozen study source snapshot changed')
    old,_=verify_baseline(root/'baseline',root/'baseline-assembly-sources')
    expected=copy.deepcopy(old['configuration']);expected['mesh']['cells']=[250,250,1]
    if p['configuration']!=expected:raise ValueError('Physical parameters differ from baseline')
    for name,digest in p['baseline_hashes'].items():
        if sha256_file(root/'baseline'/name)!=digest:raise ValueError('Imported baseline changed: '+name)
    return p
