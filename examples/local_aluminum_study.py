"""Opt-in locally refined aluminum build and resumable broadband execution.

Original uniform-grid scheduler and every solver file remain unchanged.
Preparation never launches GPU computation or a Pod. --run is explicit.
"""
import argparse,copy,hashlib,json,os,sys,time
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from examples.pzt_mesh_protocol import layout_for
from traxis.config import Material
from traxis.local_aluminum import LocalMesh,LocalAttachment

ROOT=Path(__file__).resolve().parents[1]
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def stopwatch(seconds):
    hours,rest=divmod(int(seconds),3600);minutes,seconds=divmod(rest,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

class SimulationETA:
    """Rolling propagation speed for this case/attempt, excluding startup."""
    def __init__(self,steps):self.steps=steps;self.samples=deque(maxlen=64)
    def update(self,completed,elapsed):
        if completed>=self.steps:
            return dict(eta_scope='current_simulation',eta_status='completed',eta_remaining_seconds=0.,eta_remaining='00:00:00',steps_per_second=None,estimated_finish_utc=None)
        waiting=dict(eta_scope='current_simulation',eta_status='calculating',eta_remaining_seconds=None,
            eta_remaining='Calculating...',steps_per_second=None,estimated_finish_utc=None)
        if completed<=0:
            self.samples.clear();return waiting
        if elapsed is None or not np.isfinite(elapsed) or elapsed<0:return waiting
        if self.samples and (completed<self.samples[-1][0] or elapsed<self.samples[-1][1]):self.samples.clear()
        if not self.samples or completed>self.samples[-1][0]:self.samples.append((completed,elapsed))
        # First positive progress point establishes the baseline; initial
        # compilation/capture overhead is not extrapolated over the whole run.
        if len(self.samples)<3:return waiting
        first,last=self.samples[0],self.samples[-1];dt=last[1]-first[1]
        if dt<=0:return waiting
        rate=(last[0]-first[0])/dt;remaining=(self.steps-completed)/rate
        return dict(eta_scope='current_simulation',eta_status='available',eta_remaining_seconds=remaining,
            eta_remaining=stopwatch(remaining),steps_per_second=rate,
            estimated_finish_utc=datetime.fromtimestamp(time.time()+remaining,timezone.utc).isoformat())

class RunTimers:
    """Wall time across launches; excludes time while the scheduler is stopped."""
    def __init__(self,root):
        self.path=Path(root)/'timing.json'
        self.state=json.loads(self.path.read_text()) if self.path.exists() else dict(total_elapsed_seconds=0.,cases={})
        self.total_before=self.state['total_elapsed_seconds'];self.started=time.perf_counter()
        self.case=None;self.sim_started=None
    def begin_case(self,case):
        self.case=case;self.case_started=time.perf_counter()
        previous=self.state['cases'].get(case,{})
        self.case_before=previous.get('case_elapsed_seconds',0.)
        self.sim_before=previous.get('simulation_seconds',0.);self.sim_started=None
    def begin_simulation(self):self.sim_started=time.perf_counter()
    def end_simulation(self):
        if self.sim_started is not None:self.sim_before+=time.perf_counter()-self.sim_started
        self.sim_started=None
    def checkpoint(self):
        now=time.perf_counter();self.state['total_elapsed_seconds']=self.total_before+now-self.started
        if self.case is not None:
            self.state['cases'][self.case]=dict(case_elapsed_seconds=self.case_before+now-self.case_started,
                simulation_seconds=self.sim_before+(now-self.sim_started if self.sim_started is not None else 0.))
        self.state['simulation_seconds']=sum(c['simulation_seconds'] for c in self.state['cases'].values())
        save(self.path,self.state)
        current=self.state['cases'].get(self.case,{})
        total=self.state['total_elapsed_seconds'];sim=current.get('simulation_seconds',self.state['simulation_seconds'])
        return dict(total_elapsed_seconds=total,total_stopwatch=stopwatch(total),
            case_elapsed_seconds=current.get('case_elapsed_seconds',0.),
            simulation_seconds=sim,simulation_stopwatch=stopwatch(sim),
            total_simulation_seconds=self.state['simulation_seconds'],simulation_timer_running=self.sim_started is not None)
    def end_case(self):
        self.checkpoint();self.case=None

def record_event(root,data):
    data.update(timer_updated_unix_s=time.time(),run_timer_running=data.get('status')=='running')
    save(Path(root)/'status.json',data)
    with (Path(root)/'console.log').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(data)+'\n')
    print(json.dumps(data),flush=True)

def timing_summary(root,timers,status):
    timers.checkpoint();summary=dict(status=status,**timers.state)
    save(Path(root)/'timing-summary.json',summary)
    lines=[f'Study status: {status}',f'Total run time: {stopwatch(summary["total_elapsed_seconds"])} ({summary["total_elapsed_seconds"]:.3f} s)',
        f'Total simulation time: {stopwatch(summary["simulation_seconds"])} ({summary["simulation_seconds"]:.3f} s)',
        'Times accumulate across launches, including failed attempts; time while stopped is excluded.']
    for case,values in summary['cases'].items():
        lines.append(f'{case}: case total {values["case_elapsed_seconds"]:.3f} s; simulation {values["simulation_seconds"]:.3f} s')
    (Path(root)/'TIMINGS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def source_hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('traxis','examples','configs') for p in (ROOT/folder).rglob('*')
        if p.is_file() and p.suffix in ('.py','.json')}
def local_mesh(config):
    layout=layout_for(config)
    options=config['local_aluminum_refinement']
    if not options.get('enabled'):raise ValueError('Local refinement is not enabled')
    mesh=LocalMesh(layout,Material(**config['host_material']),
        [config['actuator_center_xy_m'],config['receiver_center_xy_m']],
        patch_width_m=options['patch_width_m'],cells_per_patch_axis=options['cells_per_patch_axis'])
    return layout,mesh

def case_time_steps(plan,row):
    pair=row.get('time_steps_s')
    if pair is None:
        coarse=plan['execution'].get('fixed_coarse_dt_s') or plan['baseline_coarse_dt_s']
        pair=dict(coarse=coarse,fine=coarse/2)
    coarse=pair['coarse'];fine=pair['fine']
    if not np.isfinite([coarse,fine]).all() or coarse<=0 or fine<=0 or coarse!=2*fine:
        raise ValueError('Local study requires positive finite 2:1 coarse/fine time steps')
    return coarse,fine,round(plan['duration_s']/coarse)

def apply_time_settings(plan,options):
    plan['execution']['fixed_coarse_dt_s']=options['time_steps_s']['coarse']
    plan['substeps']=2
    for row in plan['cases']:
        row['time_steps_s']=dict(options.get('time_steps_by_rings_s',{}).get(str(row['radial']),options['time_steps_s']))
        coarse,fine,steps=case_time_steps(plan,row)
        row['estimate'].update(provisional_coarse_dt_s=coarse,provisional_fine_dt_s=fine,provisional_primary_steps=steps)
    plan['protocol']['time_policy']='Per-case time steps: six rings fine 4.9/coarse 9.8 ns; eight rings fine 4.7/coarse 9.4 ns; coupled LTS stability and temporal accuracy not certified'

def prepare(source,root):
    root=Path(root)
    if root.exists() and any(root.iterdir()):
        if (root/'cases').exists() or not (root/'plan.json').exists():raise FileExistsError('Use a new directory; existing results are preserved')
        prior=json.loads((root/'plan.json').read_text())
        if not prior.get('local_aluminum_refinement_study') or prior.get('simulation_started'):raise FileExistsError('Cannot replace an executed/unrelated plan')
        save(root/f'plan.previous-{time.time_ns()}.json',prior)
    plan=copy.deepcopy(json.loads(Path(source).read_text()))
    if plan.get('element')!='spectral':raise ValueError('This adapter preserves the existing spectral PZT study')
    options=json.loads((ROOT/'configs/local-aluminum-refinement.json').read_text())
    plan['configuration']['local_aluminum_refinement']=options
    apply_time_settings(plan,options)
    layout,mesh=local_mesh(plan['configuration'])
    # Retain all source/PZT/frequency/time settings, but never reuse kernels from another mesh.
    plan.update(local_aluminum_refinement_study=True,source_plan=str(Path(source).resolve()),
        simulation_started=False,local_aluminum_mesh=mesh.report)
    for row in plan['cases']:
        # Exact extra PZT node count from the unchanged spectral mesh geometry.
        from traxis.spectral_piezoelectric import disk_mesh
        from traxis.pzt_receiver import DiskGeometry
        c=plan['configuration'];body=disk_mesh(DiskGeometry(tuple(c['actuator_center_xy_m']),c['disk_radius_m'],c['disk_thickness_m'],'top',1),
            radial_cells=row['radial'],side_cells=row['angular']//4,layers=row['thickness'])
        free_nodes=int(np.count_nonzero(abs(body.points_m[:,2])>1e-14))
        total=3*mesh.n+9*free_nodes
        ports=[]
        for center,surface in ((c['actuator_center_xy_m'],'top'),(c['receiver_center_xy_m'],'top'),(c['receiver_center_xy_m'],'bottom')):
            nodes=body.points_m[abs(body.points_m[:,2])<1e-14].copy();nodes[:,:2]+=center
            nodes[:,2]=c['plate_m'][2] if surface=='top' else 0.
            support=set()
            for point in nodes:
                ids,_=mesh.weights(point);support.update(ids.tolist())
            ports.append(len(support))
        # Conservative separate planning envelope; measured peak is recorded at execution.
        host_storage=layout.screen['device_storage_screen_mib']*2**20+(18*3*mesh.n+6*3*mesh.old_n)*8
        execution=32*total*8+512*2**20
        row['estimate'].update(total_mechanical_dofs=total,local_aluminum_scalar_dofs=mesh.n,
            patch_dofs=[3*(free_nodes+nh) for nh in ports],local_contact_support_nodes=ports,
            estimated_gpu_peak_bytes=int(1.2*(host_storage+execution+2*2**30)),
            estimated_host_peak_bytes=int(3*2**30),
            estimated_case_total_seconds_range=[None,None],estimated_primary_kernel_seconds_range=[None,None],
            estimate_note='Conservative local tensor/PZT factor/workspace envelope; full refined runtime and peak remain unmeasured.')
    plan['local_execution']=dict(cuda_graphs=True,native_factors=True,
        integration='Original optimized LTS executor code objects with separate mesh actions',
        time_steps_changed=True,time_steps_scope='per case: cases[].time_steps_s',
        stability_qualification='Per-case requested pairs; coupled LTS stability and temporal accuracy not certified')
    plan['local_source_hashes']=source_hashes()
    save(root/'plan.json',plan);save(root/'status.json',dict(status='prepared',simulation_started=False))
    text=['Local aluminum refinement enabled: 12 mm target region, 16 target subdivisions (0.75 mm maximum spacing).',
        'Additional coarse-face alignment cuts and a surrounding transition ensure continuous displacement.',
        'Original fifth-order aluminum elements and original thickness mesh are retained.',
        'All coarse cells outside the two transition regions remain unchanged.',
        'PZT mesh/material/mass/electrical settings, broadband forcing and original LTS recurrence are unchanged.',
        'No original uniform-grid waveform is reused as a result of this different mesh.',
        'Requested time steps: six rings fine 4.9/coarse 9.8 ns; eight rings fine 4.7/coarse 9.4 ns; coupled LTS stability and temporal accuracy qualification is outstanding.',
        'Full execution requires the cloud memory budget; local GPU checks use tiny models only.',
        f'Run explicitly: python -m examples.local_aluminum_study --run --root "{root}"']
    (root/'README.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(prepared=True,full_simulations_started=0,mesh=mesh.report,
        estimated_gpu_peak_gib=[r['estimate']['estimated_gpu_peak_bytes']/2**30 for r in plan['cases']]),indent=2))
    return plan

def build(plan,row,event=lambda *a,**k:None):
    from traxis.gpu_tensor import build_gpu_tensor
    from traxis.piezoelectric import PiezoMaterial,read_profile
    from traxis.pzt_receiver import DiskGeometry
    from traxis.spectral_partial import component
    from traxis.quadratic_pzt_sparse import ConsistentMass
    from traxis.gpu_local_aluminum import build_system
    mass_backend=plan['local_execution'].get('mass_solve_backend','reference')
    if mass_backend=='batched':
        from traxis.gpu_local_aluminum_batched import build_system
    elif mass_backend!='reference':raise ValueError('Unknown mass solve backend')
    from traxis.gpu_enriched_verlet import sparse_probe_set
    import psutil,cupy as cp
    c=plan['configuration'];estimate=row['estimate']
    if estimate['estimated_gpu_peak_bytes']>.8*cp.cuda.runtime.memGetInfo()[0]:raise MemoryError('Local refined study exceeds free GPU memory')
    if estimate['estimated_host_peak_bytes']>.8*psutil.virtual_memory().available:raise MemoryError('Local refined setup exceeds free host RAM')
    layout,mesh=local_mesh(c);layout.screen['max_device_mib']=plan['protocol']['device_budget_mib']
    base=build_gpu_tensor(layout,Material(**c['host_material']))
    material=PiezoMaterial.from_profile(read_profile(ROOT/c['material_profile']))
    def geo(center,surface):return DiskGeometry(tuple(center),c['disk_radius_m'],c['disk_thickness_m'],surface,1)
    event('assembling_unchanged_pzt')
    data=component(geo(c['actuator_center_xy_m'],'top'),material,rings=row['radial'],sectors=row['angular'],layers=row['thickness'],
        max_workspace_mib=1536,max_factor_mib=1536)
    bodies=[];masses=[];next_dof=3*mesh.n
    for center,surface,electrical in [(c['actuator_center_xy_m'],'top','voltage'),
            (c['receiver_center_xy_m'],'top','open'),(c['receiver_center_xy_m'],'bottom','open')]:
        body=LocalAttachment(data,mesh,geo(center,surface),first_dof=next_dof,electrical=electrical)
        masses.append(ConsistentMass(body,np.tile(mesh.diagonal,3),
            max_factor_mib=plan['local_execution'].get('mass_factor_budget_mib',8192)));bodies.append(body);next_dof=body.next_dof
    if next_dof!=estimate['total_mechanical_dofs']:raise ArithmeticError('Prepared/local PZT DOF estimate mismatch')
    system=build_system(base,mesh,bodies,masses,native=plan['local_execution']['native_factors'])
    force=np.zeros(next_dof);force[bodies[0].indices]=-bodies[0].coupling
    probes=sparse_probe_set([(b.indices,b.coupling/b.total_capacitance_f) for b in bodies[1:]],next_dof)
    return system,force,probes

def run(root):
    from examples.pzt_broadband import interior_impulse
    from examples.pzt_mesh_study import reconstruct_stage
    from traxis.gpu_local_aluminum import fine_partition,gpu_lts_simulate
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    if source_hashes()!=plan['local_source_hashes']:raise ValueError('Prepared source/configuration snapshot changed; prepare a new study')
    if not plan['execution'].get('skip_stability_checks'):raise NotImplementedError('Retain an explicitly selected stability policy; compatible local screen not silently substituted')
    timers=RunTimers(root)
    for row in plan['cases']:
        H,fine_dt,steps=case_time_steps(plan,row)
        folder=root/'cases'/row['id'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text()).get('completed'):continue
        started=time.perf_counter();system=None;timers.begin_case(row['id']);eta=SimulationETA(steps)
        def event(phase,**extra):
            # Solver progress also includes elapsed_seconds; retain it separately.
            solver_elapsed=extra.pop('elapsed_seconds',None)
            data=dict(extra,status='failed' if phase=='failed' else 'running',case=row['id'],phase=phase,
                elapsed_seconds=time.perf_counter()-started,fine_dt_s=fine_dt,coarse_dt_s=H,coarse_steps=steps,
                **timers.checkpoint())
            if solver_elapsed is not None:data['solver_elapsed_seconds']=solver_elapsed
            if 'steps_completed' in data:data['percent']=100*data['steps_completed']/steps
            if phase in ('propagating','simulation_finished','configuration_saved'):
                data.update(eta.update(data.get('steps_completed',steps),solver_elapsed))
            else:data.update(eta_scope='current_simulation',eta_status='unavailable',eta_remaining_seconds=None,eta_remaining='--')
            record_event(root,data)
        try:
            event('building_local_aluminum');system,force,probes=build(plan,row,event)
            save(folder/'build.json',dict(screen=system.screen,local_mesh=system.local_mesh.report))
            fine=fine_partition(system,force,plan.get('boxes',()),plan.get('edge_width_m',0.))
            stage=folder/'primary';stage.mkdir(exist_ok=True)
            if not (stage/'result.json').exists():
                timers.begin_simulation();event('propagating',steps_completed=0,requested_steps=steps)
                try:
                    t,y,stats=gpu_lts_simulate(system,steps,H,probes,force,fine=fine,force=interior_impulse(H),
                        source_sampling=plan['source_sampling'],screen_stability=False,use_graphs=plan['local_execution']['cuda_graphs'],
                        max_workspace_mib=plan['protocol']['device_budget_mib'],progress=lambda s:event('propagating',**s))
                finally:timers.end_simulation()
                event('simulation_finished',steps_completed=steps,requested_steps=steps)
                np.savez_compressed(stage/'trace.npz',time_s=t,voltage_v=y)
                save(stage/'result.json',dict(completed=True,statistics=stats,time_steps_s=row.get('time_steps_s',dict(coarse=H,fine=fine_dt)),
                    simulation_seconds=timers.checkpoint()['simulation_seconds'],
                    sha256=hashlib.sha256((stage/'trace.npz').read_bytes()).hexdigest()))
            receipt=json.loads((stage/'result.json').read_text())
            if receipt['sha256']!=hashlib.sha256((stage/'trace.npz').read_bytes()).hexdigest():raise ValueError('Saved local-mesh kernel checksum mismatch')
            reconstruct_stage(folder,'primary',plan['protocol']['frequencies_hz'])
            save(folder/'result.json',dict(completed=True,seconds=time.perf_counter()-started,
                timing=timers.checkpoint(),source_hashes=plan['local_source_hashes']))
            event('configuration_saved')
        except BaseException as exc:
            save(folder/'failure.json',dict(error=str(exc),completed=False,timing=timers.checkpoint()))
            event('failed',error=str(exc));timing_summary(root,timers,'failed');raise
        finally:
            try:
                if system is not None:system.close()
            finally:
                timers.end_case();timing_summary(root,timers,'failed' if sys.exc_info()[0] is not None else 'running')
    timing_summary(root,timers,'completed')
    record_event(root,dict(status='completed',phase='study_finished',pod_action='none',**timers.checkpoint()))

def main():
    p=argparse.ArgumentParser(__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--source-plan',type=Path);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if a.prepare:
        if a.source_plan is None:p.error('--source-plan required for preparation')
        prepare(a.source_plan,a.root)
    else:run(a.root)
if __name__=='__main__':main()
