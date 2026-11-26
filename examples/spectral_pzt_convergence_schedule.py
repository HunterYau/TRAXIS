"""Sequential, restartable local Q2 radial component study (6 and 8 rings)."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RINGS = (6, 8)


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu','gpu'], default='cpu')
    parser.add_argument('--local-cuda-root', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import psutil
    from examples.spectral_pzt_experiment import compare
    reference_path = ROOT / 'docs/evidence/spectral-pzt/q2-s16-r5-z1-g6.json'
    previous = json.loads(reference_path.read_text())
    assert previous['status'] == 'completed'
    schedule = dict(order=2,side_cells=16,layers=1,quadrature_order=6,backend=args.backend,
        frequencies_hz=[50000,100000,150000,200000,250000,300000],modes=['S0','A0'],
        precision='float64',mass='full consistent',reference=str(reference_path),
        comparison='Complex voltage change; denominator is the preceding mesh. No fitted phase or gain.',
        scope='Prescribed-motion PZT component only; no full plate simulation.',
        available_ram_mib=psutil.virtual_memory().available/2**20,
        estimated_resources='Several seconds to several minutes per case; allow roughly 1-3 GiB process RAM. Sparse factor fill is not measured for these meshes.',
        cases=[dict(radial_cells=r,nodes=(33**2+256*r)*3,
                    hexahedra=256+64*r) for r in RINGS])
    save(output/'schedule.json',schedule)
    report = dict(status='running',cases=[],schedule=schedule)
    started = time.perf_counter()
    save(output/'summary.json',report)
    print('Q2 CONVERGENCE: 6 rings, then 8 rings. Saved 5-ring reference.',flush=True)
    print('Output:',output,flush=True)
    print('Available RAM: %.0f MiB' % schedule['available_ram_mib'],flush=True)
    try:
        for index, radial in enumerate(RINGS,1):
            print('\nConfiguration %d/2: %d rings, all six frequencies, S0 and A0' % (index,radial),flush=True)
            case_output=output/('rings-%d'%radial)
            case_output.mkdir(exist_ok=True)
            suffix='-gpu' if args.backend=='gpu' else ''
            result_path=case_output/('q2-s16-r%d-z1-g6%s.json'%(radial,suffix))
            if result_path.exists() and json.loads(result_path.read_text()).get('status')=='completed':
                print('Reusing completed configuration.',flush=True)
            else:
                # Preserve incomplete attempts and start a separate attempt folder.
                if result_path.exists():
                    case_output=case_output/('attempt-'+time.strftime('%Y%m%dT%H%M%S'))
                    case_output.mkdir()
                    result_path=case_output/('q2-s16-r%d-z1-g6%s.json'%(radial,suffix))
                code="import sys,runpy; sys.path.insert(0,'.'); runpy.run_module('examples.spectral_pzt_experiment',run_name='__main__')"
                command=[sys.executable,'-u','-c',code,'--order','2','--side','16',
                    '--radial',str(radial),'--layers','1','--quadrature','6',
                    '--max-equations','30000','--output',str(case_output),'--backend',args.backend]
                if args.local_cuda_root is not None:
                    command+=['--local-cuda-root',str(args.local_cuda_root)]
                process=subprocess.Popen(command,cwd=ROOT)
                last=time.monotonic()
                while process.poll() is None:
                    time.sleep(1)
                    if time.monotonic()-last>=10:
                        try:
                            child=psutil.Process(process.pid)
                            print('  Still computing; RAM %.0f MiB; total elapsed %.1f s' %
                                  (child.memory_info().rss/2**20,time.perf_counter()-started),flush=True)
                        except psutil.NoSuchProcess:
                            pass
                        last=time.monotonic()
                if process.returncode:
                    raise RuntimeError('Component process failed with exit code '+str(process.returncode))
            result=json.loads(result_path.read_text())
            row=dict(radial_cells=radial,result_file=str(result_path),result=result)
            report['cases'].append(row)
            if result['status']!='completed':
                raise RuntimeError('Configuration did not complete: '+result['status'])
            difference=compare(previous,result)
            row['vs_previous']=dict(reference_id=previous['id'],metrics=difference)
            print('\nRESULT: %d rings; %.2f seconds; peak RAM %.1f MiB' %
                  (radial,result['seconds'],result['sampled_peak_rss_mib']),flush=True)
            for mode in ['S0','A0']:
                metrics=difference[mode]
                print(mode+' aggregate voltage change: %.6f%%'%metrics['relative_l2_percent'],flush=True)
                print(' kHz | voltage change % | phase change degrees | below 1% | below 0.1%',flush=True)
                metrics['below_1_percent']=[v<1 for v in metrics['per_frequency_percent']]
                metrics['below_0_1_percent']=[v<.1 for v in metrics['per_frequency_percent']]
                for f,error,phase in zip(schedule['frequencies_hz'],metrics['per_frequency_percent'],metrics['phase_change_deg']):
                    print('%4d | %16.6f | %20.6f | %s | %s'%(f//1000,error,phase,error<1,error<.1),flush=True)
            save(output/'summary.json',report)
            previous=result
        report['status']='completed'
        print('\nBoth configurations completed. Results saved.',flush=True)
    except BaseException as exc:
        report.update(status='failed_or_interrupted',error=repr(exc))
        print('Study stopped:',repr(exc),flush=True)
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        save(output/'summary.json',report)


if __name__=='__main__':
    main()
