"""Exploratory, unfitted packet timing against measured PZT channels."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import hilbert
from examples.experimental_five import MATERIAL
from traxis.dispersion import mode_velocity
from traxis.storage import sha256_file, write_json

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'outputs/wandowski-pzt-plate1'


def packet(t, signal, frequency, mode):
    common = np.arange(1750)*.2e-6
    if t[0] > common[0] or t[-1] < common[-1]:
        raise ValueError('incomplete shared interval')
    y = np.interp(common, t, signal)
    y -= y[common < 20e-6].mean()
    envelope = abs(hilbert(y, 8192)[:len(y)])
    flight = .25/mode_velocity(frequency, .001, mode, MATERIAL).group_m_s
    selected = np.flatnonzero((common >= flight) & (common <= flight+5/frequency))
    peak = selected[np.argmax(envelope[selected])]
    return dict(packet_peak_us=float(common[peak]*1e6),
                flight_us=float((common[peak]-2.5/frequency)*1e6),
                window_us=[flight*1e6,(flight+5/frequency)*1e6],
                window_edge=bool(peak in (selected[0],selected[-1])))


def main():
    records=[]
    for frequency in (50000,100000,150000,200000,250000):
        directory=ROOT/f'{frequency//1000}khz/native'
        result=json.loads((directory/'result.json').read_text())
        path=directory/'waveforms.npz'
        if not result['completed'] or sha256_file(path)!=result['waveforms_sha256']:
            raise ValueError('incomplete or altered simulation output')
        with np.load(path) as data:
            for mode in ('S0','A0'):
                for channel,name in enumerate(('top','bottom')):
                    real=packet(data['experimental_time_s'],data['experimental_pzt_voltage_v'][:,channel],frequency,mode)
                    sim=packet(data['simulation_time_s'],data['ideal_open_reference_voltage_v'][:,channel],frequency,mode)
                    delta=sim['flight_us']-real['flight_us']
                    records.append(dict(frequency_hz=frequency,mode=mode,channel=name,
                        real=real,simulation=sim,delay_us=delta,
                        signed_flight_difference_percent=100*delta/real['flight_us'],
                        absolute_flight_difference_percent=100*abs(delta)/real['flight_us'],
                        waveform_sha256=sha256_file(path)))
    evidence=dict(method='Independent top/bottom PZT channels; 5 MHz common grid, baseline first 20 us, Hilbert envelope with 8192 padding; fixed theoretical group-arrival to arrival+five-cycle windows; subtract half burst duration. No fitted shift, gain or polarity/mode sums.',
        limitations=['Packet windows can contain overlapping modes/reflections; peaks are timing proxies, not verified mode labels.',
                     'Window-edge maxima are unreliable.',
                     'Percentages describe flight-time difference, not overall accuracy.',
                     'Ideal readout and omitted PZT/glue dynamics remain; no absolute voltage validation.'],
        analysis_sha256=sha256_file(Path(__file__)),records=records)
    target=PROJECT/'docs/evidence/pzt-measured-timing-comparison-20261117.json'
    write_json(target,evidence)
    for row in records:
        print(json.dumps(row))


if __name__=='__main__':main()
