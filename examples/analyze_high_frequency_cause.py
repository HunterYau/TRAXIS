"""Analyze saved data and closed-form equations only; never run a simulation."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import periodogram, find_peaks

from examples.experimental_five import record
from traxis.config import Material
from traxis.piezoelectric import PiezoMaterial, read_profile
from traxis.storage import write_json, sha256_file
from traxis.tensor_hex import line_rule


def spectral_peaks(t, voltage, lower=5.05e6, upper=5.45e6):
    mask=(t>=150e-6)&(t<=350e-6)
    f,p=periodogram((voltage[:,0]-voltage[:,1])[mask],fs=1/(t[1]-t[0]),
                    window='hann',nfft=262144)
    peaks,_=find_peaks(p)
    peaks=sorted((k for k in peaks if lower<f[k]<upper),key=lambda k:p[k],reverse=True)[:7]
    result=[]
    for k in peaks:
        log=np.log(p[k-1:k+2])
        offset=.5*(log[0]-log[2])/(log[0]-2*log[1]+log[2])
        result.append(dict(frequency_hz=float(f[k]+offset*(f[1]-f[0])),psd=float(p[k])))
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--saved-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((args.root/'plan.json').read_text());config=plan['configuration']
    modes=[];sources=[]
    for name in ('baseline','time-2','time-4'):
        folder=args.root/'250khz'/name
        r=json.loads((folder/'result.json').read_text());path=folder/'waveforms.npz'
        if not r['completed'] or sha256_file(path)!=r['waveforms_sha256']:raise ValueError(path)
        with np.load(path) as data:
            t=data['time_s'];v=data['voltage_v'];peaks=spectral_peaks(t,v)
        dt=config['mesh']['duration_s']/r['case']['steps']
        for peak in peaks:
            # Inverse of the exact Verlet numerical-frequency relation.
            # This is a mesh-system frequency, NOT a measured real resonance.
            peak['inferred_mesh_system_hz']=float(np.sin(np.pi*dt*peak['frequency_hz'])/(np.pi*dt))
        modes.append(dict(case=name,dt_s=dt,stored_sample_rate_hz=float(1/(t[1]-t[0])),
                          peaks=peaks))
        sources.append(dict(path=str(path),sha256=sha256_file(path)))
    reference=modes[-1]['peaks'][0]['inferred_mesh_system_hz']
    for mode in modes:
        predicted=float(np.arcsin(np.pi*reference*mode['dt_s'])/(np.pi*mode['dt_s']))
        mode.update(predicted_dominant_hz=predicted,
                    observed_minus_predicted_hz=mode['peaks'][0]['frequency_hz']-predicted,
                    predicted_extra_cycles_over_350us=(predicted-reference)*.00035)
    observed=[m['peaks'][0]['frequency_hz'] for m in modes]
    shift_ratio=(observed[0]-observed[1])/(observed[1]-observed[2])

    drives=[];frequency_sweep=[]
    for f in (50000,100000,150000,200000,250000):
        row=record(f);drive=np.asarray(row['nifgen_avg_waveform_ch1'])
        t=np.arange(len(drive))/row['sampleRate'];duration=row['numCycles']/f
        analytic=np.where(t<=duration,np.sin(2*np.pi*f*t)*np.sin(np.pi*t/duration)**2,0.)
        drives.append(dict(frequency_hz=f,max_abs_sample_difference_v=float(np.max(abs(drive-analytic))),
            relative_sample_difference=float(np.linalg.norm(drive-analytic)/np.linalg.norm(drive))))
        folder=args.saved_run/f'{f//1000}khz';path=folder/'waveforms.npz'
        r=json.loads((folder/'result.json').read_text())
        if not r['completed'] or r['waveforms_sha256']!=sha256_file(path):raise ValueError(path)
        with np.load(path) as data:
            peaks=spectral_peaks(data['simulation_time_s'],data['simulated_receiver_voltage_v'],4.5e6,5.5e6)
        frequency_sweep.append(dict(carrier_hz=f,predicted_upper_source_image_hz=row['sampleRate']+f,
                                    dominant_late_receiver_hz=peaks[0]['frequency_hz']))
        sources.append(dict(path=str(path),sha256=sha256_file(path)))

    row=record(250000);drive=np.asarray(row['nifgen_avg_waveform_ch1'])
    t=np.arange(len(drive))/row['sampleRate'];grid=np.arange(32768)*(.0001/32768)
    duration=row['numCycles']/250000
    analytic=np.where(grid<=duration,np.sin(2*np.pi*250000*grid)*np.sin(np.pi*grid/duration)**2,0.)/max(abs(drive))
    linear=np.interp(grid,t,drive/max(abs(drive)),left=0.,right=0.)
    input_spectra={}
    for name,signal in (('linear',linear),('analytic_definition',analytic)):
        f,p=periodogram(signal,fs=1/(grid[1]-grid[0]))
        input_spectra[name]=dict(image_to_carrier_power_db=float(10*np.log10(
            p[(f>=5.1e6)&(f<=5.4e6)].sum()/p[(f>=150000)&(f<=350000)].sum())))

    profile=Path('configs/pzt/nce51-noliac-ver1404.json');data=read_profile(profile)
    material=PiezoMaterial.from_profile(data);c,e,eps=material.stress_charge
    cD=c+e.T@np.linalg.solve(eps,e);thickness=config['disk_thickness_m']
    thickness_scales=dict(manufacturer_nominal_thickness_hz=2000/thickness,
        constrained_bulk_short_scale_hz=float(np.sqrt(c[2,2]/material.density_kg_m3)/(2*thickness)),
        constrained_bulk_open_scale_hz=float(np.sqrt(cD[2,2]/material.density_kg_m3)/(2*thickness)),
        limit='Scale estimates, not eigenfrequencies of the bonded discrete system; no new modal solve.')
    host=Material(**config['host_material']);lam,mu=host.lame
    gap=float(np.max(np.diff(line_rule(config['mesh']['order'])[0]))*config['plate_m'][0]/config['mesh']['cells'][0]/2)
    spatial=dict(maximum_in_plane_node_gap_m=gap,
        host_bulk_longitudinal_wavelength_m=float(np.sqrt((lam+2*mu)/host.density_kg_m3)/reference),
        host_bulk_shear_wavelength_m=float(np.sqrt(mu/host.density_kg_m3)/reference),
        limit='Resolution warning from bulk reference scales; not a high-frequency plate dispersion or mesh convergence proof.')
    write_json(args.output/'cause-analysis.json',dict(sources=sources,modes=modes,
        dominant_frequency_shift_ratio=shift_ratio,reference_inferred_mesh_system_hz=reference,
        drive_sample_checks=drives,input_spectra=input_spectra,frequency_sweep=frequency_sweep,
        thickness_scales=thickness_scales,spatial_scales=spatial,
        simulation_runs_started=0,solver_or_physical_parameters_changed=False,
        inference='Saved modal peak shifts closely follow Verlet temporal dispersion. Input interpolation creates '
        'source images near the troublesome band. Causal origin and physical identity of those modes remain unconfirmed.',
        limits=['Spectral peak interpolation does not increase the 200-us window information/resolution.',
                'Fine-run inferred frequency is a diagnostic reference, not independent exact modal truth.',
                'No phase alignment or spectral warping was applied to validation waveforms.',
                'No new temporal, modal, spatial or source-reconstruction simulation was set up or run.']))
    print('Dominant observed frequencies MHz:',[f/1e6 for f in observed])
    print('Frequency-shift ratio (expected about 4):',shift_ratio)
    print('Predicted frequency residual Hz:',[m['observed_minus_predicted_hz'] for m in modes])
    print('Input spectra:',input_spectra)
    print('Thickness scales:',thickness_scales)
    print('Spatial scales:',spatial)


if __name__=='__main__':main()
