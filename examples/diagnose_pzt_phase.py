"""Audit saved voltages for phase mismatch without modifying simulation results.

Fitted rotations/delays are diagnostic only, never validation scores or physics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
from scipy.optimize import minimize_scalar

from traxis.signal_comparison import waveform_metrics
from traxis.storage import sha256_file, write_json


def phase_diagnostic(measured, simulated):
    """Least-squares quadrature fit; reports how much constant phase can explain."""
    measured=np.asarray(measured, float); simulated=np.asarray(simulated, float)
    raw=waveform_metrics(measured, simulated)
    if min(np.linalg.norm(measured),np.linalg.norm(simulated))==0:
        return dict(raw=raw, diagnostic_phase_fit=None, phase_rotation_degrees=None,
                    fitted_gain=None)
    quadrature=hilbert(simulated).imag
    design=np.column_stack((simulated, quadrature))
    coefficients=np.linalg.lstsq(design, measured, rcond=None)[0]
    fitted=design@coefficients
    return dict(raw=raw,
        diagnostic_phase_fit=waveform_metrics(measured, fitted),
        phase_rotation_degrees=float(np.degrees(np.arctan2(-coefficients[1],coefficients[0]))),
        fitted_gain=float(np.linalg.norm(coefficients)))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]; sources=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,2,figsize=(12,13))
    for i,khz in enumerate((50,100,150,200,250)):
        directory=args.root/f'{khz}khz'
        path=directory/'waveforms.npz'
        result=json.loads((directory/'result.json').read_text())
        if not result['completed'] or sha256_file(path)!=result['waveforms_sha256']:
            raise ValueError('Incomplete or changed waveform: '+str(path))
        comparison=result['comparison']
        sources.append(dict(path=str(path),sha256=sha256_file(path)))
        grid=np.arange(1750)*2e-7
        with np.load(path) as data:
            real=np.column_stack([np.interp(grid,data['experimental_time_s'],data['experimental_pzt_voltage_v'][:,j]) for j in range(2)])
            sim=np.column_stack([np.interp(grid,data['simulation_time_s'],data['simulated_receiver_voltage_v'][:,j]) for j in range(2)])
        real-=real[grid<20e-6].mean(axis=0);sim-=sim[grid<20e-6].mean(axis=0)
        for j,(label,sign,mode) in enumerate((('fast',1,'S0'),('slow',-1,'A0'))):
            r=real[:,0]+sign*real[:,1];s=sim[:,0]+sign*sim[:,1]
            detail=comparison[label+'_details']
            # Union of independently located packet bounds, with no score-based gate selection.
            windows=comparison[label+'_adaptive_window']
            lower=min(windows[k]['window_us'][0] for k in ('measured','simulated'))
            upper=max(windows[k]['window_us'][1] for k in ('measured','simulated'))
            mask=(grid*1e6>=lower)&(grid*1e6<=upper)
            row=dict(frequency_khz=khz,trace=mode,arrival_status=comparison[label+'_status'],
                interval_us=[lower,upper],full_record=phase_diagnostic(r,s),packet=phase_diagnostic(r[mask],s[mask]))
            # Independent time-shift diagnostic within half a carrier period.
            bound=.5/(khz*1000)
            def score(delay):
                shifted=np.interp(grid[mask]-delay,grid,s,left=0,right=0)
                return waveform_metrics(r[mask],shifted)['normalized_wiggle_difference_percent']
            delays=np.linspace(-bound,bound,401)
            best=int(np.argmin([score(d) for d in delays]))
            fit=minimize_scalar(score,bounds=(delays[max(0,best-1)],delays[min(400,best+1)]),
                method='bounded',options={'xatol':1e-12})
            row['packet_delay_fit']=dict(added_delay_us=float(fit.x*1e6),difference_percent=float(fit.fun),
                search_bound_us=bound*1e6,at_bound=bool(abs(fit.x)>.99*bound))
            row['packet_peak_delay_us']=detail['measured']['packet_peak_us']-detail['simulated']['packet_peak_us']
            rows.append(row)
            ax=axes[i,j];t=grid[mask]*1e6
            ax.plot(t,r[mask]/np.linalg.norm(r[mask]),label='Measured',lw=1)
            ax.plot(t,s[mask]/np.linalg.norm(s[mask]),label='Simulated',lw=1)
            ax.set_title(f'{khz} kHz {mode}-related trace (unshifted)');ax.grid(alpha=.2)
            if i==0:ax.legend()
            if i==4:ax.set_xlabel('Time (microseconds)')
    fig.tight_layout();fig.savefig(args.output/'unshifted-packets.png',dpi=140);plt.close(fig)
    write_json(args.output/'diagnostic.json',dict(sources=sources,records=rows,
        limitations='Fits are diagnostic only: no physical calibration, no altered validation scores. '
        'Sum/difference are mode-related traces, not proven pure modes. '
        'Constant phase and delay are alternative hypotheses, not uniquely identified causes. '
        'Packet quadrature is computed on the cropped interval; boundary effects remain possible.'))
    lines=['kHz mode | packet wiggle difference: original -> phase fit -> delay fit | phase deg | delay us']
    for row in rows:
        p=row['packet'];d=row['packet_delay_fit']
        lines.append(f"{row['frequency_khz']} {row['trace']} | "
            f"{p['raw']['normalized_wiggle_difference_percent']:.2f} -> "
            f"{p['diagnostic_phase_fit']['normalized_wiggle_difference_percent']:.2f} -> "
            f"{d['difference_percent']:.2f} | {p['phase_rotation_degrees']:.1f} | {d['added_delay_us']:.3f}")
    text='\n'.join(lines)+'\n'
    (args.output/'summary.txt').write_text(text,encoding='utf-8');print(text)


if __name__=='__main__':main()
