"""Plot retained raw mechanical waveforms and differences from a sealed study."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from examples.embedded_wave import checked_case
from examples.embedded_wave_study import PROTOCOL
from examples.tensor_wave import PROJECT, archive_schema, checked_receipt, load_arrays, read_json, sealed
from traxis.storage import sha256_file, write_json


def plot(study, output):
    study,output=Path(study).resolve(),Path(output).resolve()
    if not output.is_relative_to(PROJECT/'outputs'):
        raise ValueError('refinement figures must remain in project outputs')
    reports=[p for p in study.glob('*-refinement-report.json') if p.name in
        ('spatial-refinement-report.json','time-refinement-report.json')]
    if len(reports)!=1:raise ValueError('figure input needs exactly one retained refinement report')
    report=checked_receipt(read_json(reports[0]));mode=report['mode']
    if (not report['completed'] or report['protocol']!=PROTOCOL or mode not in ('spatial','time')
            or report['ready_for_training_data'] or report['training_updates']!=0
            or report['analysis_sha256']!=sha256_file(PROJECT/'examples/embedded_wave_study.py')):
        raise ValueError('figure input study is incomplete or its analysis changed')
    n=PROTOCOL['comparison_intervals']+1
    spec=dict(time_s=(n,),traces_m=(3,n,3))
    archive_schema(study/'comparison.npz',report['comparison_sha256'],spec)
    arrays=load_arrays(study/'comparison.npz',spec)
    time_s,traces=arrays['time_s'],arrays['traces_m']
    if not np.array_equal(time_s,np.arange(n)*(PROTOCOL['duration_s']/(n-1))):
        raise ValueError('figure input physical time grid changed')
    norms=np.linalg.norm(traces[-1],axis=0)
    differences=[np.linalg.norm(traces[i]-traces[i+1],axis=0) for i in (0,1)]
    if len(report['receivers'])!=3:raise ValueError('figure needs the three declared receiver results')
    for i,row in enumerate(report['receivers']):
        actual=[differences[0][i],differences[1][i],norms[i]]
        declared=[row[k] for k in ('coarse_difference_l2_m','fine_difference_l2_m','finest_waveform_l2_m')]
        change_matches=(row['finest_relative_change'] is None if norms[i]==0 else
            row['finest_relative_change'] is not None and np.isclose(row['finest_relative_change'],actual[1]/norms[i],rtol=8*np.finfo(float).eps,atol=0))
        if (row['receiver']!=i or not np.allclose(actual,declared,rtol=8*np.finfo(float).eps,atol=0)
                or not change_matches):
            raise ValueError('figure input metrics disagree with retained raw traces')
    case=checked_case(read_json(Path(report['levels'][0]['directory'])/'embedded-wave-plan.json')['case'])
    output.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(output/'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=[f'Q{p}' for p in PROTOCOL['spatial_orders']] if mode=='spatial' else [f'{steps:,} steps' for steps in PROTOCOL['time_steps']]
    colors=['#767676','#0072B2','#D55E00']
    fig,axes=plt.subplots(2,3,figsize=(13.2,7.2),sharex=True,layout='constrained')
    for i in range(3):
        top,bottom=axes[:,i];row=report['receivers'][i]
        for j,label in enumerate(labels):
            top.plot(time_s*1e6,traces[j,:,i]*1e9,label=label,color=colors[j],linewidth=.85,alpha=.9)
        for j in (0,1):
            label=f'{labels[j]} minus {labels[j+1]}' if mode=='spatial' else ('Coarse minus middle','Middle minus fine')[j]
            bottom.plot(time_s*1e6,(traces[j,:,i]-traces[j+1,:,i])*1e9,
                label=label,color=colors[j+1],linewidth=.8)
        top.set_title(f'Receiver {i+1} at {tuple(case["receiver_centers_xy_m"][i])} m',fontsize=10)
        change='undefined (zero trace)' if row['finest_relative_change'] is None else f'{row["finest_relative_change"]*100:.5g}%'
        note=f'Finest change: {change} | '+('PASS' if row['passed'] else 'FAIL')
        if mode=='time':
            order='unidentified' if row['observed_time_order'] is None else f'{row["observed_time_order"]:.6f}'
            note+=f' | order {order}'
        bottom.set_title(note,fontsize=9)
        bottom.set_xlabel('Physical time (microseconds)')
        for ax in (top,bottom):
            ax.axhline(0,color='#999999',linewidth=.5,zorder=0);ax.grid(alpha=.16);ax.set_xlim(0,500)
        top.set_ylabel('Mean normal displacement (nm)')
        bottom.set_ylabel('Raw displacement difference (nm)')
    trace_handles,trace_labels=axes[0,0].get_legend_handles_labels()
    difference_handles,difference_labels=axes[1,0].get_legend_handles_labels()
    fig.legend(trace_handles+difference_handles,
        ['Trace: '+label for label in trace_labels]+['Difference: '+label for label in difference_labels],
        loc='outside lower center',ncol=5,fontsize=8,frameon=False)
    state='Recessed' if case['defective'] else 'Intact'
    status='PASS' if report['passed'] else 'FAIL'
    fig.suptitle(f'{state} 500 x 500 x 3.175 mm plate: {mode} refinement — {status}\n'
        '100 kHz mechanical disk source; unchanged raw samples; numerical check only',fontsize=12)
    fig.savefig(output/'refinement.png',dpi=180)
    fig.savefig(output/'refinement.pdf',metadata={'Creator':'TRAXIS'})
    plt.close(fig)
    receipt=sealed(dict(completed=True,passed=True,study_passed=report['passed'],mode=mode,
        study_report_sha256=sha256_file(reports[0]),study_fingerprint=report['fingerprint'],
        comparison_sha256=report['comparison_sha256'],plotter_sha256=sha256_file(__file__),
        files={name:sha256_file(output/name) for name in ('refinement.png','refinement.pdf')},
        scope='Visualizes retained mechanical numerical convergence; no physical or PZT validation.'))
    write_json(output/'figure-report.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(plot(args.study,args.output)))


if __name__=='__main__':main()
