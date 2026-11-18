"""Apply the paper's PZT addition/subtraction, without fitted channel gains."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.compare_pzt_recordings import packet, PROJECT, ROOT
from traxis.storage import sha256_file, write_json


def separate(channels):
    channels=np.asarray(channels)
    if channels.ndim!=2 or channels.shape[1]!=2 or not np.isfinite(channels).all():
        raise ValueError('finite top/bottom pair required')
    return {'S0':channels[:,0]+channels[:,1], 'A0':channels[:,0]-channels[:,1]}


def main():
    t=np.arange(1750)*2e-7
    records=[]
    fig,axes=plt.subplots(5,2,figsize=(14,15),sharex=True)
    for index,f in enumerate((50000,100000,150000,200000,250000)):
        folder=ROOT/f'{f//1000}khz/native'; path=folder/'waveforms.npz'
        result=json.loads((folder/'result.json').read_text())
        if not result['completed'] or sha256_file(path)!=result['waveforms_sha256']:
            raise ValueError('saved simulation identity mismatch')
        with np.load(path) as data:
            pairs=[]
            for tk,yk in [('experimental_time_s','experimental_pzt_voltage_v'),
                          ('simulation_time_s','ideal_open_reference_voltage_v')]:
                if data[tk][0]>t[0] or data[tk][-1]<t[-1]:raise ValueError('incomplete interval')
                channels=np.column_stack([np.interp(t,data[tk],data[yk][:,j]) for j in range(2)])
                channels-=channels[t<20e-6].mean(axis=0)
                pairs.append(separate(channels))
            for column,mode in enumerate(('S0','A0')):
                real,sim=[p[mode] for p in pairs]
                observed=packet(t,real,f,mode); predicted=packet(t,sim,f,mode)
                error=100*abs(predicted['flight_us']-observed['flight_us'])/observed['flight_us']
                er=abs(hilbert(real,8192)[:len(t)]); es=abs(hilbert(sim,8192)[:len(t)])
                edge=observed['window_edge'] or predicted['window_edge']
                # The manuscript reports detectable direct A0 only up to 140 kHz
                # for these PZTs; do not reinterpret residual peaks as confirmed A0.
                status='exploratory_timing' if not edge else 'window_edge_unreliable'
                if mode=='A0' and f>140000:status='above_reported_A0_detection_range'
                row=dict(frequency_hz=f,mode=mode,operation='top+bottom' if mode=='S0' else 'top-bottom',
                    experimental=observed,simulation=predicted,raw_absolute_flight_difference_percent=error,
                    reportable_timing_difference_percent=error if status=='exploratory_timing' else None,
                    status=status,waveform_sha256=sha256_file(path),
                    full_record_envelope_correlation=float(np.corrcoef(er,es)[0,1]))
                records.append(row)
                ax=axes[index,column]
                ax.plot(t*1e6,er/er.max(),label='Measured',lw=1.2)
                ax.plot(t*1e6,es/es.max(),label='Simulated',lw=1.2)
                ax.axvspan(*observed['window_us'],alpha=.1,color='green')
                ax.set_title(f'{f//1000} kHz | {row["operation"]} | {status}',fontsize=9)
                ax.set_xlim(0,350);ax.set_ylim(0,1.1);ax.grid(alpha=.2)
                if index==0:ax.legend()
                if index==4:ax.set_xlabel('Time (microseconds)')
                ax.set_ylabel('Normalized envelope')
    fig.suptitle('Paper method: add for fast wave, subtract for slow wave\nNo channel gain, polarity, or time-shift fitting; green = timing search',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95))
    out=PROJECT/'docs/evidence'
    plot=out/'pzt-paired-comparison-20261117.png';fig.savefig(plot,dpi=140);plt.close(fig)
    evidence=dict(source='https://zenodo.org/records/18937842',method='Section 5 equations 13/14: unscaled top+bottom and top-bottom. Raw measured channel signs retained; simulation uses declared +z signs. First-20-us baseline, common 5 MHz grid, entire 350-us record plotted. Same fixed packet windows and half-burst flight definition as previous comparison.',
        limitations=['Not the separate mode-indicator method.', 'No per-channel amplitude equalization or polarity fit.',
                     'Paper reports direct A0 detectable only up to 140 kHz with these PZTs.',
                     'Addition/subtraction suppresses opposite symmetry, not same-mode reflections.',
                     'Timing percentages are exploratory; no overall accuracy or absolute voltage claim.'],
        analysis_sha256=sha256_file(Path(__file__)),packet_analysis_sha256=sha256_file(PROJECT/'examples/compare_pzt_recordings.py'),
        figure_sha256=sha256_file(plot),records=records)
    write_json(out/'pzt-paired-comparison-20261117.json',evidence)
    for row in records:print(json.dumps(row))


if __name__=='__main__':main()
