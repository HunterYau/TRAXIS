"""Report saved SEM component tests; no simulation or operator execution."""
import os
from pathlib import Path
import hashlib,json,shutil
import numpy as np
from examples.spectral_pzt_experiment import compare

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'outputs/spectral-pzt-experiment';dest=ROOT/'docs/evidence/spectral-pzt'
    dest.mkdir(parents=True,exist_ok=True)
    ref=json.loads((source/'saved-reference.json').read_text());cases=[];previous=None
    for radial in range(1,6):
        path=source/f'q2-s16-r{radial}-z1-g6.json';data=json.loads(path.read_text())
        row=dict(id=data['id'],result=data,vs_tetra_reference=compare(ref,data),
                 vs_previous=None if previous is None else compare(previous,data))
        cases.append(row);previous=data
    q3=json.loads((source/'q3-s16-r1-z1-g6-f150-300.json').read_text())
    cases.append(dict(id=q3['id'],result=q3,vs_tetra_reference=compare(ref,q3)))
    six=json.loads((source/'q2-s16-r2-z1-g6.json').read_text())
    eight=json.loads((source/'q2-s16-r2-z1-g8.json').read_text())
    t=json.loads((source/'timing-repeat/tetra/r6-a64-t2.json').read_text())
    h=json.loads((source/'timing-repeat/q2-s16-r2-z1-g6.json').read_text())
    actions=json.loads((source/'actions/action-check.json').read_text())
    a,b=[sum(v['timings'].values()) for v in actions['cases']]
    report=dict(reference=ref,reference_source='docs/evidence/quadratic-pzt-dimensions/comparisons.json:reference',
        cases=cases,quadrature6_vs8=compare(six,eight),reference_repeat_difference=compare(ref,t),
        timings=dict(tetra=t,hex=h,harmonic_component_speedup=t['seconds']/h['seconds'],
                     gpu_action_pair_speedup=a/b,actions=actions),
        q3_initial_attempt=json.loads((source/'q3-resource-stop.json').read_text()),
        interpretation='Isolated prescribed-motion PZT component comparison only. Same 64-sided polygon and full consistent mass. No plate/LTS/GPU full-run speed or true accuracy claim. Q2 closest match is not proven converged.')
    (dest/'comparison.json').write_text(json.dumps(report,indent=2))
    for path in source.glob('*.json'):shutil.copy2(path,dest/path.name)
    for path in source.glob('q3-*.log'):shutil.copy2(path,dest/path.name)
    shutil.copy2(source/'actions/action-check.json',dest/'gpu-actions.json')
    os.environ['MPLCONFIGDIR']=str(source/'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    frequencies=np.array([50,100,150,200,250,300])
    for ax,mode in zip(axes,['S0','A0']):
        for row in cases[:5]:
            ax.plot(frequencies,row['vs_tetra_reference'][mode]['per_frequency_percent'],marker='o',label='Q2 radial '+str(row['result']['radial_cells']))
        ax.axhline(1,color='gray',linestyle='--',label='1% difference')
        ax.set(title=mode+' voltage difference from P2 6/64/2',xlabel='Frequency (kHz)',ylabel='Complex voltage difference (%)')
        ax.grid(alpha=.25)
    axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(dest/'voltage-differences.png',dpi=160);plt.close(fig)
    original=ROOT.parent/'pzt-quadratic-tetrahedra';proof={}
    for folder in ['traxis','examples','tests','configs']:
        for path in (original/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.json'):
                relative=path.relative_to(original);target=ROOT/relative
                if not target.exists():raise AssertionError('Missing preserved source '+str(relative))
                old=hashlib.sha256(path.read_bytes()).hexdigest();new=hashlib.sha256(target.read_bytes()).hexdigest()
                if old!=new:raise AssertionError('Original quadratic source modified '+str(relative))
                proof[relative.as_posix()]=old
    (dest/'preserved-source-sha256.json').write_text(json.dumps(dict(all_original_sources_identical=True,files=proof),indent=2))
    print('CPU harmonic speedup:',report['timings']['harmonic_component_speedup'])
    print('GPU action pair speedup:',report['timings']['gpu_action_pair_speedup'])
    print('Original source files preserved:',len(proof))
    print('Report:',dest/'comparison.json')


if __name__=='__main__':main()
