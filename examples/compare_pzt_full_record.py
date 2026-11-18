"""Full-record PZT comparison, with no fitted shifts or arrival windows."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import hilbert, find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from traxis.storage import sha256_file, write_json

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'outputs/wandowski-pzt-plate1'
OUT=PROJECT/'docs/evidence'


def signal_metrics(real, sim, frequency):
    # Same zero padding and time origin for both recordings. No fitted lag,
    # amplitude, time warp, or channel polarity is used for these metrics.
    er=abs(hilbert(real,8192)[:len(real)])
    es=abs(hilbert(sim,8192)[:len(sim)])
    corr=float(np.corrcoef(er,es)[0,1])
    voltage_corr=float(np.corrcoef(real,sim)[0,1])
    def peaks(e):
        ids,_=find_peaks(e,prominence=.02*e.max(),distance=max(1,int(.5/frequency/2e-7)))
        return [dict(time_us=float(i*.2),relative_height=float(e[i]/e.max())) for i in ids]
    ir=int(np.argmax(er)); iss=int(np.argmax(es))
    return dict(envelope_pearson_r=corr,voltage_pearson_r=voltage_corr,
        measured_global_peak_us=ir*.2,simulated_global_peak_us=iss*.2,
        global_peak_delay_us=(iss-ir)*.2,
        global_peak_time_difference_percent=100*abs(iss-ir)/ir if ir else None,
        measured_peaks=peaks(er),simulated_peaks=peaks(es),
        record_edge_peak=bool(ir in (0,len(er)-1) or iss in (0,len(es)-1))),er,es


def main():
    common=np.arange(1750)*2e-7
    fig,axes=plt.subplots(5,2,figsize=(15,16),sharex=True)
    records=[]
    for row,frequency in enumerate((50000,100000,150000,200000,250000)):
        folder=ROOT/f'{frequency//1000}khz/native'
        result=json.loads((folder/'result.json').read_text())
        path=folder/'waveforms.npz'
        if not result['completed'] or sha256_file(path)!=result['waveforms_sha256']:
            raise ValueError('result or waveform identity check failed')
        with np.load(path) as data:
            for channel,name in enumerate(('top','bottom')):
                traces=[]
                for tk,yk in [('experimental_time_s','experimental_pzt_voltage_v'),
                              ('simulation_time_s','ideal_open_reference_voltage_v')]:
                    t=data[tk]; y=data[yk][:,channel]
                    if t[0]>common[0] or t[-1]<common[-1]:raise ValueError('incomplete interval')
                    signal=np.interp(common,t,y)
                    signal-=signal[common<20e-6].mean()
                    if not np.isfinite(signal).all() or np.std(signal)==0:raise ValueError('invalid trace')
                    traces.append(signal)
                metrics,er,es=signal_metrics(*traces,frequency)
                records.append(dict(frequency_hz=frequency,channel=name,**metrics,
                                    waveform_sha256=sha256_file(path)))
                ax=axes[row,channel]
                ax.plot(common*1e6,er/er.max(),label='Measured envelope',lw=1.5)
                ax.plot(common*1e6,es/es.max(),label='Simulated envelope',lw=1.2,alpha=.85)
                ax.set_title(f'{frequency//1000} kHz, {name}: envelope r={metrics["envelope_pearson_r"]:.3f}')
                ax.set_xlim(0,350); ax.set_ylim(0,1.12); ax.grid(alpha=.2)
                ax.set_ylabel('Individually normalized envelope')
                if row==0:ax.legend(fontsize=9)
                if row==4:ax.set_xlabel('Time from excitation start (microseconds)')
    fig.suptitle('Full available PZT record: no time shifts or mode-selection windows\nNormalization for display only; later peaks can be reflections',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96))
    image=OUT/'pzt-full-record-comparison-20261117.png'
    fig.savefig(image,dpi=140);plt.close(fig)
    evidence=dict(interval_us=[0,349.8],simulation_duration_us=350,sampling_hz=5000000,
        method='Full shared record, baseline first 20 us; Hilbert envelope with 8192 padding. Pearson correlation on entire envelope and voltage records, no fitted lag/gain/polarity. Global peaks scanned over entire record; local maxima use 2% prominence and half-cycle minimum spacing.',
        limitations=['Correlation is not a percentage accuracy or acceptance score.',
                     'Global maxima can represent different modes/reflections; their timing difference is not automatically direct-wave travel-time error.',
                     'Top/bottom evaluated separately; no inferred wiring signs.',
                     'No calibrated absolute voltage comparison; ideal electrical load and missing PZT/glue dynamics remain.'],
        analysis_sha256=sha256_file(Path(__file__)),figure_sha256=sha256_file(image),records=records)
    write_json(OUT/'pzt-full-record-comparison-20261117.json',evidence)
    for r in records:
        print(json.dumps({k:v for k,v in r.items() if k not in ('measured_peaks','simulated_peaks','waveform_sha256')}))


if __name__=='__main__':main()
