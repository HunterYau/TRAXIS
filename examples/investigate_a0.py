"""Saved-data A0 spectrum and drive-image investigation; no full simulations."""
import argparse
from fractions import Fraction
from pathlib import Path
import json
import numpy as np
from scipy.signal import periodogram,resample_poly, butter,sosfiltfilt
from examples.experimental_five import record
from examples.pzt_convergence import metrics
from examples.pzt_drive_reconstruction import reconstructed_drive
from traxis.storage import write_json,sha256_file


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    saved=[];sources=[]
    for name in ('baseline','time-2','time-4'):
        p=args.root/'250khz'/name/'waveforms.npz'
        result=json.loads((p.parent/'result.json').read_text())
        if not result['completed'] or sha256_file(p)!=result['waveforms_sha256']:raise ValueError(p)
        with np.load(p) as d:saved.append((d['time_s'].copy(),d['voltage_v'].copy()))
        sources.append(dict(path=str(p),sha256=sha256_file(p)))
    t=saved[0][0];fs=1/(t[1]-t[0]);rows=[]
    for i,j in ((0,1),(1,2)):
        a,b=saved[i][1],saved[j][1]
        for lo,hi in ((0,350),(40,85),(90,135),(150,350)):
            mask=(t*1e6>=lo)&(t*1e6<=hi)
            entry=dict(pair=[i,j],interval_us=[lo,hi],raw=metrics(a[mask],b[mask]),spectra={})
            for mode,sign in (('S0',1),('A0',-1)):
                diff=((a[:,0]+sign*a[:,1])-(b[:,0]+sign*b[:,1]))[mask]
                f,p=periodogram(diff,fs=fs)
                entry['spectra'][mode]=dict(peak_hz=float(f[np.argmax(p)]),
                    difference_power_above_500khz_percent=float(100*p[f>500000].sum()/p.sum()))
            rows.append(entry)
        # Declared diagnostics only: do not replace raw full-band convergence.
        sos=butter(8,500000,fs=fs,output='sos')
        rows.append(dict(pair=[i,j],study='500 kHz low-pass diagnostic, not a physical scope model',
                         channels=metrics(sosfiltfilt(sos,a,axis=0),sosfiltfilt(sos,b,axis=0))))
        ratio=Fraction(5000000/fs).limit_denominator(100000)
        target=np.arange(1750)/5000000
        def sample(x):
            y=resample_poly(x,ratio.numerator,ratio.denominator,axis=0,window=('kaiser',10.))
            return np.column_stack([np.interp(target,np.arange(len(y))/5000000,y[:,c]) for c in range(2)])
        rows.append(dict(pair=[i,j],study='anti-aliased 5 MHz sampling diagnostic, not identified NI filter',
                         channels=metrics(sample(a),sample(b))))
    row=record(250000);drive=np.asarray(row['nifgen_avg_waveform_ch1']);dt=1/row['sampleRate']
    drive_t=np.arange(len(drive))*dt
    reconstruct,info=reconstructed_drive(drive_t,drive)
    grid=np.arange(32768)*(.0001/32768);sample_fs=1/(grid[1]-grid[0])
    methods={'linear':np.interp(grid,drive_t,drive/max(abs(drive)),left=0,right=0),
             'reconstructed_candidate':reconstruct(grid)}
    drive_report={}
    for name,signal in methods.items():
        f,p=periodogram(signal,fs=sample_fs)
        base=p[(f>=150000)&(f<=350000)].sum()
        image=p[(f>=5100000)&(f<=5400000)].sum()
        drive_report[name]=dict(image_to_base_power_db=float(10*np.log10(image/base)))
    write_json(args.output/'investigation.json',dict(sources=sources,records=rows,
        drive_image=drive_report,candidate=info,
        limits='A0-related subtraction is not a proven pure mode. Fixed packet intervals are diagnostic. '
        'Spectral coincidence suggests a source-image/high-frequency ringing mechanism but does not prove causality. '
        'Neither diagnostic filtering nor candidate drive reconstruction proves full-band convergence or experimental fidelity.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    for i,j in ((0,1),(1,2)):
        d=saved[i][1]-saved[j][1];f,p=periodogram(d[:,0]-d[:,1],fs=fs)
        ax.semilogy(f/1e6,np.maximum(p,1e-35),label=f'level {i} vs {j}')
    ax.axvline(.25,color='k',ls=':',label='250 kHz excitation')
    ax.set(xlabel='Frequency (MHz)',ylabel='A0-related difference PSD (V²/Hz)',xlim=(0,7))
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(args.output/'difference-spectrum.png',dpi=150)
    print('Drive image:',drive_report)
    for r in rows:
        if 'channels' in r:print(r['pair'],r['study'],{k:v['raw_l2_change_percent'] for k,v in r['channels'].items()})


if __name__=='__main__':main()
