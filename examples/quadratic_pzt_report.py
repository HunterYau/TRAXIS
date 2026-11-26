"""Summarize existing isolated tests, without starting calculations."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def response(row,mode):
    return np.array([complex(x['voltage_real_v'],x['voltage_imag_v']) for x in row['rows'] if x['mode']==mode])

def differences(ref,candidate):
    result={}
    for mode in ('S0','A0'):
        a,b=response(ref,mode),response(candidate,mode);errors=100*abs(b-a)/abs(a)
        result[mode]=dict(per_frequency_percent=errors.tolist(),worst_percent=float(max(errors)),
                          relative_l2_percent=float(100*np.linalg.norm(b-a)/np.linalg.norm(a)))
    return result

def main():
    root=ROOT/'outputs/quadratic-pzt-audit'
    radial=[json.loads(p.read_text()) for p in (root/'benchmark').glob('*.json')]
    base=[json.loads(p.read_text()) for p in (root/'basewaves').glob('*.json')]
    lookup={r['id']:r for r in base};ref=lookup['r6-a64-t2']
    rows=[]
    for case in base:
        if case['status']=='completed':rows.append(dict(id=case['id'],nodes=case['nodes'],seconds=case['seconds'],rss_mib=case['sampled_peak_rss_mib'],metrics=differences(ref,case)))
    checks={
        'radial_5_to_6':differences(lookup['r6-a64-t2'],lookup['r5-a64-t2']),
        'angular_64_to_96_at_radial4_thickness2':differences(lookup['r4-a96-t2'],lookup['r4-a64-t2']),
        'thickness_2_to_3_at_radial4_angular64':differences(lookup['r4-a64-t3'],lookup['r4-a64-t2'])}
    report=dict(reference='r6-a64-t2, numerical reference, not exact ground truth',frequencies_hz=[50000,100000,150000,200000,250000,300000],
        cases=rows,independent_checks=checks,radial_free_body_tests=radial,basewave_receipts=base,
        interpretation='P2 passes polynomial/assembly tests. A0 remains sensitive near cancellation; no all-frequency 1% or 0.1% convergence claim.',
        preferred_next_exploratory_start='r5-a64-t2; promising overall, not accepted as converged',
        restrictions='No plate transient or RunPod. Prescribed base motion omits plate backreaction; polygon changes when angular subdivisions change. Timings used mixed concurrent CPU load.')
    (root/'results.json').write_text(json.dumps(report,indent=2))
    lines=['QUADRATIC PZT ISOLATED TESTS','Branch research/pzt-quadratic-tetrahedra','References: finest radial run 6/64/2; angular 4/64/2 vs 4/96/2; thickness 4/64/2 vs 4/64/3.',
           'Errors below compare complex harmonic receiver voltages with no fitted gain, phase, or delay.','Not full plate waveforms, not Wandowski comparisons, no temporal stepping.',
           'Mesh | nodes | S0 L2 % | A0 L2 % | S0 worst % | A0 worst % | seconds | sampled RSS MiB']
    for r in sorted(rows,key=lambda r:r['nodes']):
        s,a=r['metrics']['S0'],r['metrics']['A0']
        lines.append(f"{r['id']} | {r['nodes']} | {s['relative_l2_percent']:.4f} | {a['relative_l2_percent']:.4f} | {s['worst_percent']:.4f} | {a['worst_percent']:.4f} | {r['seconds']:.2f} | {r['rss_mib']:.1f}")
    lines+=['','5/64/2 is a next experimental candidate, not an optimum or a proven converged mesh.','Its full six-frequency response L2 differences are below 1%, but A0 pointwise differences are not.','The 7/64/2 run was not assembled: exceeded the conservative 8192-node/2.5GiB diagnostic forecast budget. This does not prove it cannot fit on this computer.','All inherited physics/LTS files must retain start-of-task SHA256 values.','Original linear implementation remains the default and fallback. No experimental backend enabled in the existing scheduler.']
    (root/'results.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for mode,ax in zip(('S0','A0'),axes):
        for id in ['r2-a64-t1','r3-a64-t2','r4-a64-t2','r5-a64-t2']:
            ax.semilogy(np.arange(50,301,50),differences(ref,lookup[id])[mode]['per_frequency_percent'],marker='o',label=id)
        ax.axhline(1,color='black',linestyle='--',linewidth=1);ax.set(title=f'{mode}: prescribed surface wave',xlabel='Frequency (kHz)',ylabel='Complex voltage difference (%)');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Isolated quadratic PZT tests against 6/64/2; no plate transient')
    fig.tight_layout();fig.savefig(root/'frequency-errors.png',dpi=160);plt.close(fig)
    print(root/'results.txt')

if __name__=='__main__':main()
