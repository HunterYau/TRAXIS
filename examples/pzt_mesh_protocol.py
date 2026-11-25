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
from traxis.pzt_receiver import DiskGeometry,_point_weights
from traxis.storage import sha256_file,write_json,canonical_hash

PROJECT=Path(__file__).resolve().parents[1]
CONFIG=PROJECT/'configs/pzt-mesh-convergence'
BASE=dict(radial=6,angular=32,thickness=3)
LEVELS=[dict(radial=8,angular=40,thickness=4),dict(radial=10,angular=48,thickness=5),
        dict(radial=12,angular=64,thickness=6)]


def cases():
    rows=[]
    for level,values in enumerate(LEVELS,1):
        for family in ('radial','angular','thickness','combined'):
            resolution=dict(BASE);resolution.update(values if family=='combined' else {family:values[family]})
            order=len(rows)+1
            rows.append(dict(order=order,id=f'{order:02d}-{family}-L{level}',family=family,level=level,**resolution))
    return rows


def fingerprints():
    paths=[*sorted((PROJECT/'traxis').glob('*.py')),*sorted((PROJECT/'examples').glob('*.py')),
           *sorted(CONFIG.glob('*.json')),PROJECT/'configs/pzt/nce51-noliac-ver1404.json']
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
    mesh,face=disk_mesh(geometry,sectors=s,rings=r,layers=z,max_nodes=8192,max_elements=65536)
    points=mesh.points_m;tets=mesh.tetrahedra
    jac=(points[tets[:,1:]]-points[tets[:,0,None]]).transpose(0,2,1)
    singular=np.linalg.svd(jac,compute_uv=False)
    if np.any(singular[:,-1]<=0) or np.max(singular[:,0]/singular[:,-1])>1e8:raise ValueError('Invalid mesh quality')
    patches=[]
    for center,surface in ((config['actuator_center_xy_m'],'top'),(config['receiver_center_xy_m'],'top'),
                           (config['receiver_center_xy_m'],'bottom')):
        contact=points[:face,:2]+np.array(center);host=set()
        for xy in contact:
            ids,_=_point_weights(layout,(*xy,config['plate_m'][2] if surface=='top' else 0.))
            nz=layout.cells[2]*layout.order+1
            ids=ids[ids%nz==(nz-1 if surface=='top' else 0)]
            host.update(ids.tolist())
        patches.append(3*len(host)+3*face*z)
    ndof=layout.screen['vector_dofs']+9*face*z
    squares=sum(n*n for n in patches);retained=2*squares*8
    attachment=max((12*(3*len(points))**2+10*n*n)*8 for n in patches)
    merge=10*squares*8
    host=1.25*(max(attachment+retained,merge+retained)+48*ndof*8)
    storage=layout.screen['device_storage_screen_mib']*1024**2+3*squares*8+24*ndof
    gpu=storage+max(48*ndof*8+3*squares*8,32*ndof*8)
    return dict(body_nodes_each=len(points),body_tetrahedra_each=len(tets),
        added_mechanical_dofs=9*face*z,total_mechanical_dofs=ndof,patch_dofs=patches,
        min_tetra_singular_m=float(singular[:,-1].min()),max_tetra_condition=float((singular[:,0]/singular[:,-1]).max()),
        attachment_workspace_bytes=attachment,merge_workspace_bytes=merge,
        estimated_host_peak_bytes=int(host),estimated_gpu_peak_bytes=int(1.15*gpu),
        estimate_note='Dense consistent PZT matrices retained; forecasts, not measured peaks; runtime rechecks actual available memory.')


def prepare(root,baseline,assembly_sources):
    root=Path(root)
    if root.exists():raise FileExistsError('Use a new study output directory')
    old,compatibility=verify_baseline(baseline,assembly_sources)
    protocol=json.loads((CONFIG/'protocol.json').read_text())
    rows=cases()
    for row in rows:
        if json.loads((CONFIG/(row['id']+'.json')).read_text())!=row:raise ValueError('Configuration table changed')
    c=copy.deepcopy(old['configuration'])
    c['mesh']['cells']=[old['case']['cells'],old['case']['cells'],old['case'].get('nz',1)]
    if c.get('setup_profile') or c.get('damping_profile') or c['body']['bond']!='perfect':
        raise ValueError('Unexpected physical setup or damping profile')
    base=geometric_estimate(c,dict(BASE,id='baseline'))
    stable=None
    for line in (Path(baseline)/'console.log').read_text().splitlines():
        try:
            data=json.loads(line)
            if data.get('phase')=='bodies_ready':stable=data['stable_dt_s']
        except ValueError:pass
    if stable is None:raise ValueError('Missing recorded baseline stability limit')
    build_seconds=json.loads((Path(baseline)/'build.json').read_text())['seconds']
    solve_seconds=json.loads((Path(baseline)/'lts-kernel/result.json').read_text())['seconds']
    for row in rows:
        e=geometric_estimate(c,row);ratio=min(1.,e['min_tetra_singular_m']/base['min_tetra_singular_m'])
        H=old['coarse_dt_s'];k=0
        while H/2>.8*stable*ratio:H/=2;k+=1
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
    plan=dict(schema=1,scope='PZT mesh convergence, intact plate, unchanged LTS split-LFC broadband integration',
        protocol=protocol,cases=rows,configuration=c,baseline_hashes=hashes,baseline_compatibility=compatibility,
        baseline_coarse_dt_s=old['coarse_dt_s'],baseline_coarse_steps=old['coarse_steps'],duration_s=.00035,
        edge_width_m=old['edge_width_m'],boxes=old['boxes'],source_sampling='coarse',substeps=2,
        identity=fingerprints(),baseline_rerun=False,simulation_started=False,
        temporal_comparison='Each mesh primary vs half step; target 0.01% for raw/envelope/wiggle. Failed checks remain unresolved; no automatic extra meshes or optimal selection.',
        baseline_limit='Saved baseline has no mesh-specific half-step check; comparison against it cannot alone certify 0.1% spatial accuracy.')
    write_json(root/'plan.json',plan);write_json(root/'status.json',dict(status='prepared',completed=[],failed=[]))
    lines=['PZT mesh convergence: baseline reused; no baseline run.',
        'Order: level 1 then 2 then 3; radial, angular, thickness, combined at each level.',
        '12 meshes, 24 mandatory impulse solves (primary and half-step); six reconstructions per solve.',
        'Counts and geometry exact; memory/runtime are forecasts. Actual stable steps calculated on RunPod.',
        'Order | Case | radial/angular/thickness | DOFs | GPU forecast GiB | RAM forecast GiB | provisional primary/half steps | case minutes range']
    for row in rows:
        e=row['estimate'];lo,hi=e['estimated_case_total_seconds_range']
        lines.append(f"{row['order']:02d} | {row['id']} | {row['radial']}/{row['angular']}/{row['thickness']} | {e['total_mechanical_dofs']} | {e['estimated_gpu_peak_bytes']/1024**3:.1f} | {e['estimated_host_peak_bytes']/1024**3:.1f} | {e['provisional_primary_steps']}/{e['provisional_half_step_steps']} | {lo/60:.1f}-{hi/60:.1f}")
    (root/'schedule.txt').write_text('\n'.join(lines)+'\n')
    print((root/'schedule.txt').read_text(),flush=True)
    return plan


def validate(root,*,sources=True):
    root=Path(root);p=json.loads((root/'plan.json').read_text())
    if p['schema']!=1 or p['baseline_rerun'] is not False or p['substeps']!=2 or p['source_sampling']!='coarse':
        raise ValueError('Study protocol changed')
    actual=[{k:v for k,v in row.items() if k!='estimate'} for row in p['cases']]
    if actual!=cases():raise ValueError('Exactly the 12 requested spatial configurations are required')
    if p['protocol']!=json.loads((CONFIG/'protocol.json').read_text()):raise ValueError('Protocol changed')
    if sources and p['identity']!=fingerprints():raise ValueError('Frozen study source snapshot changed')
    old,_=verify_baseline(root/'baseline',root/'baseline-assembly-sources')
    expected=copy.deepcopy(old['configuration']);expected['mesh']['cells']=[250,250,1]
    if p['configuration']!=expected:raise ValueError('Physical parameters differ from baseline')
    for name,digest in p['baseline_hashes'].items():
        if sha256_file(root/'baseline'/name)!=digest:raise ValueError('Imported baseline changed: '+name)
    return p
