"""Compare saved one-dimension refinements. No new component solves."""
import json,hashlib
from pathlib import Path
import numpy as np
from examples.quadratic_pzt_report import response

ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'outputs/quadratic-pzt-dimensions'
    schedule=json.loads((folder/'schedule.json').read_text());path=ROOT/schedule['baseline_path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=schedule['baseline_sha256']:raise ValueError('Saved reference changed')
    baseline=json.loads(path.read_text());results=[]
    for case in schedule['cases']:
        ident=f"r{case['radial']}-a{case['angular']}-t{case['thickness']}";p=folder/(ident+'.json')
        if not p.exists():raise FileNotFoundError('Refinement has not finished: '+ident)
        tested=json.loads(p.read_text());r=dict(configuration=case,status=tested['status'],receipt=tested)
        if tested['status']=='completed':
            metrics={}
            for mode in ['S0','A0']:
                a,b=response(baseline,mode),response(tested,mode)
                if len(a)!=6 or len(b)!=6:raise ValueError('Six frequency results required')
                error=100*abs(b-a)/abs(a);phase=np.degrees(np.angle(b/a))
                metrics[mode]=dict(relative_l2_percent=float(100*np.linalg.norm(b-a)/np.linalg.norm(a)),
                    worst_relative_percent=float(max(error)),per_frequency_percent=error.tolist(),
                    magnitude_change_percent=(100*(abs(b)/abs(a)-1)).tolist(),phase_change_deg=phase.tolist(),
                    absolute_voltage_difference_v=abs(b-a).tolist())
            r['metrics']=metrics
        results.append(r)
    report=dict(reference=baseline,reference_sha256=schedule['baseline_sha256'],reference_reused=True,
        frequencies_hz=[50000,100000,150000,200000,250000,300000],results=results,
        method='Complex harmonic voltage difference; no fitted gain, phase, or delay',
        limitations='Isolated PZT with prescribed plate-surface displacement. No plate backreaction or LTS time stepping. Angular changes also alter polygon boundary. One perturbation per dimension does not prove convergence.')
    (folder/'comparisons.json').write_text(json.dumps(report,indent=2))
    lines=['QUADRATIC PZT: INDEPENDENT DIMENSION REFINEMENT','Saved reference 6/64/2 reused without rerun. All six frequencies; both S0 and A0.',
        'Differences are relative to the saved reference, not true errors.','Dimension | configuration | S0 L2 % | A0 L2 % | S0 maximum % | A0 maximum % | seconds | sampled peak RSS MiB']
    for r in results:
        c=r['configuration'];d=r['receipt']
        if r['status']!='completed':lines.append(f"{c['dimension']} | {r['status']}");continue
        s,a=r['metrics']['S0'],r['metrics']['A0']
        lines.append(f"{c['dimension']} | {c['radial']}/{c['angular']}/{c['thickness']} | {s['relative_l2_percent']:.5f} | {a['relative_l2_percent']:.5f} | {s['worst_relative_percent']:.5f} | {a['worst_relative_percent']:.5f} | {d['seconds']:.2f} | {d['sampled_peak_rss_mib']:.1f}")
        lines.append('  A0 per-frequency %: '+', '.join(f'{x:.4f}' for x in a['per_frequency_percent']))
    lines+=['','No full plate simulation was run. Uniform material, forcing scale, and electrical conditions remain fixed.','Angular changes include polygon geometry effects; no new optimum or converged mesh is inferred.']
    (folder/'comparisons.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for mode,ax in zip(['S0','A0'],axes):
        for r in results:
            if r['status']=='completed':ax.semilogy(report['frequencies_hz'],r['metrics'][mode]['per_frequency_percent'],marker='o',label=r['configuration']['dimension'])
        ax.set(title=mode,xlabel='Frequency (Hz)',ylabel='Complex voltage difference (%)');ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Independent PZT refinements versus saved 6/64/2; isolated receiver')
    fig.tight_layout();fig.savefig(folder/'dimension-errors.png',dpi=160);plt.close(fig)
    print('\n'.join(lines))

if __name__=='__main__':main()
