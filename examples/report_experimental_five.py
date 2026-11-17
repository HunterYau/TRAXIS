"""Verify and record each of the five completed experimental timing comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np

from examples.experimental_five import ROOT, PROJECT, FREQUENCIES, arrival
from traxis.storage import sha256_file, write_json, canonical_hash


def report(frequency):
    if frequency not in FREQUENCIES:
        raise ValueError('frequency is outside the five-test batch')
    index=FREQUENCIES.index(frequency)+1
    directory=ROOT/f'trial-{index}-{frequency//1000}khz'
    native=directory/'native'
    measured=json.loads((native/'result.json').read_text(encoding='utf-8'))
    monitor=json.loads((directory/'launcher.json').read_text(encoding='utf-8'))
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    if (not measured['completed'] or not monitor['completed'] or monitor['exit_code']!=0
            or measured['protocol_sha256']!=sha256_file(ROOT/'protocol.json')
            or measured['source_sha256']!=protocol['source_sha256']
            or measured['frequency_hz']!=frequency
            or measured['statistics']['steps_completed']!=protocol['steps']):
        raise ValueError('trial, source identity or completion check failed')
    for name,key in (('waveforms.npz','waveforms_sha256'),('final-state.npz','final_state_sha256')):
        if sha256_file(native/name)!=measured[key]:
            raise ValueError('saved wave data changed')
    if measured['cpu_force_relative_error']>1e-10 or measured['post_burst_modified_energy_relative_drift']>1e-8:
        raise ValueError('trial numerical checks failed')
    with np.load(native/'waveforms.npz',allow_pickle=False) as wave:
        predicted=(wave['top_axial_strain']-wave['bottom_axial_strain'])/2
        observed=(wave['experimental_top_fbg_v']-wave['experimental_bottom_fbg_v'])/2
        checks=(arrival(wave['simulation_time_s'],predicted,frequency),
                arrival(wave['experimental_time_s'],observed,frequency))
        if checks!=(measured['simulation'],measured['experimental']):
            raise ValueError('arrival re-extraction disagrees with saved result')
        plot(directory,wave,predicted,observed,measured)
    keys=('frequency_hz','experimental','simulation','arrival_error_us','flight_relative_error',
          'cpu_force_relative_error','post_burst_modified_energy_relative_drift','cooling_seconds',
          'wall_seconds','protocol_sha256','waveforms_sha256','final_state_sha256','resource_peaks')
    compact={k:measured[k] for k in keys}
    compact.update(trial=index,completed=True,ready_for_training_data=False,
        conclusion='exploratory A0 arrival timing only; full transducer response unvalidated',
        peaks_inside_extraction_windows=not any(c['peak_at_window_edge'] for c in checks),
        monitor_resource_peaks=monitor['resource_peaks'],
        native_result_sha256=sha256_file(native/'result.json'),
        launcher_sha256=sha256_file(directory/'launcher.json'),
        worker_log_sha256=sha256_file(directory/'worker.log'),
        plot_sha256=sha256_file(directory/'comparison.png'),
        analysis_source_sha256=sha256_file(Path(__file__)))
    compact['fingerprint']=canonical_hash(compact)
    evidence=PROJECT/f'docs/evidence/experimental-trial-{index}-{frequency//1000}khz-20261116.json'
    # Portable signals and plots belong in the research record. Large final
    # displacement/velocity fields remain local, with their hashes recorded.
    for source,suffix in ((native/'waveforms.npz','.npz'),(directory/'comparison.png','.png')):
        destination=evidence.with_suffix(suffix)
        if destination.exists():
            if sha256_file(destination)!=sha256_file(source):
                raise ValueError('portable evidence bytes changed')
        else:destination.write_bytes(source.read_bytes())
    if evidence.exists():
        if json.loads(evidence.read_text(encoding='utf-8'))!=compact:
            raise ValueError('do not overwrite changed evidence')
    else:write_json(evidence,compact)
    return compact


def plot(directory,wave,predicted,observed,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,5.8),sharex=True,layout='constrained')
    for axis,t,y,kind,color,reference in (
        (axes[0],wave['experimental_time_s'],observed,'Measured FBG voltage (V)','#186a84',result['experimental']),
        (axes[1],wave['simulation_time_s'],predicted,'Predicted axial strain','#a75b24',result['simulation'])):
        y=y-y[t<20e-6].mean()
        axis.plot(t*1e6,y,color=color,lw=.9)
        lo,hi=reference['window_s']
        axis.axvspan(lo*1e6,hi*1e6,color='#dadada',alpha=.45)
        axis.axvline(reference['packet_peak_s']*1e6,color='black',ls='--',lw=.9)
        axis.set_ylabel(kind);axis.grid(alpha=.2)
    lo,hi=result['experimental']['window_s']
    axes[-1].set_xlim(max(0,lo*1e6-20),hi*1e6+20)
    axes[-1].set_xlabel('Time from generator signal start (microseconds)')
    fig.suptitle(f"{result['frequency_hz']//1000} kHz: A0 timing difference {result['arrival_error_us']:+.2f} microseconds\n"
                 'One aluminum plate; ideal actuator approximation; amplitudes have different units',fontsize=11)
    fig.savefig(directory/'comparison.png',dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('frequency',type=int)
    args=parser.parse_args()
    result=report(args.frequency)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
