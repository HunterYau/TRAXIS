"""Read-only plan checks; optional tiny GPU check, never full propagation."""
import argparse,json,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))

def validate(root):
    from examples.local_aluminum_study import source_hashes,case_time_steps
    plan=json.loads((Path(root)/'plan.json').read_text())
    if not plan.get('local_aluminum_refinement_study'):raise ValueError('Expected local aluminum study')
    if source_hashes()!=plan['local_source_hashes']:raise ValueError('Source/configuration checksum mismatch')
    if [r['radial'] for r in plan['cases']]!=[6,8]:raise ValueError('Expected six-ring then eight-ring schedule')
    if plan['protocol']['frequencies_hz']!=[50000,100000,150000,200000,250000,300000]:raise ValueError('Frequency schedule changed')
    for row,want in zip(plan['cases'],[(9.8e-9,4.9e-9),(9.4e-9,4.7e-9)]):
        coarse,fine,steps=case_time_steps(plan,row)
        if (coarse,fine)!=want or steps!=row['estimate']['provisional_primary_steps']:raise ValueError('Time plan mismatch')
    return plan

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',action='store_true');a=p.parse_args()
    plan=validate(a.root);report=dict(plan_valid=True,configurations=2,full_simulations_started=0,
        time_steps=[r['time_steps_s'] for r in plan['cases']],
        estimated_peak_gpu_gib=max(r['estimate']['estimated_gpu_peak_bytes'] for r in plan['cases'])/2**30,
        coupled_lts_stability_and_temporal_accuracy_certified=False)
    if a.gpu:
        import platform,psutil,cupy as cp,subprocess
        if sys.platform!='linux' or platform.machine()!='x86_64' or sys.version_info[:2]!=(3,12):raise ValueError('Use Linux x86_64 Python 3.12')
        if cp.cuda.runtime.runtimeGetVersion()!=12090:raise ValueError('Use pinned CUDA 12.9 runtime')
        free,total=cp.cuda.runtime.memGetInfo()
        from examples.pzt_mesh_study import available_host_bytes
        host=available_host_bytes()
        for row in plan['cases']:
            if row['estimate']['estimated_gpu_peak_bytes']>.8*free:raise MemoryError('Insufficient free GPU memory for prepared estimate')
            if row['estimate']['estimated_host_peak_bytes']>.8*host:raise MemoryError('Insufficient free host memory for prepared estimate')
        check_module='examples.local_aluminum_batched_check' if plan['local_execution'].get('mass_solve_backend')=='batched' else 'examples.local_aluminum_check'
        subprocess.run([sys.executable,'-m',check_module,'--host-order','5',
            '--output',str(a.root/'runpod-component-preflight')],cwd=PROJECT,check=True)
        report.update(free_gpu_gib=free/2**30,free_host_gib=host/2**30,tiny_gpu_graph_check_passed=True,
            checked_mass_backend=plan['local_execution'].get('mass_solve_backend','reference'))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
