"""Five frozen, exploratory plate-propagation comparisons with public measurements.

The disk actuator uses ideal radial edge traction. FBGs use mean axial strain.
Neither approximation includes transducer mass, glue, electronics or voltage
calibration. Results measure arrival-time agreement, not full PZT validation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.io import loadmat
from scipy.linalg import eigh
from scipy.signal import hilbert

from traxis.config import Material, Plate
from traxis.dispersion import mode_velocity
from traxis.gpu_tensor import make_layout, build_gpu_tensor, streamed_action
from traxis.gpu_enriched_verlet import gpu_explicit_simulate, sparse_probe_set
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import atomic_write, sha256_file, write_json
from traxis.tensor_hex import line_rule
from traxis.tensor_patches import _basis
from examples.embedded_wave import ThermalPause
from traxis.elmer_reference import run_monitored

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'outputs/experimental-validation-20261116'
FREQUENCIES = (50000, 100000, 150000, 200000, 250000)
MATERIAL = Material(young_pa=71e9, poisson=.33, density_kg_m3=2700.)
PLATE = Plate(1., 1., .001)
CELLS = (250, 250, 1)
ORDER = 5
STEPS = 65536
DURATION = .00035


class PeakGuard(ResourceGuard):
    """Keep all-time peaks even when the bounded sample history is full."""
    def enforce(self, sample, gpus):
        if not hasattr(self, 'peaks'):
            self.peaks = dict(system_memory_percent=0.,process_rss_mib=0.,
                              gpu_memory_mib=0.,gpu_temperature_c=0.,minimum_available_mib=math.inf)
        for key in ('system_memory_percent','process_rss_mib'):
            self.peaks[key]=max(self.peaks[key],sample[key])
        self.peaks['minimum_available_mib']=min(self.peaks['minimum_available_mib'],sample['available_mib'])
        for gpu in gpus:
            self.peaks['gpu_memory_mib']=max(self.peaks['gpu_memory_mib'],gpu['used_mib'])
            self.peaks['gpu_temperature_c']=max(self.peaks['gpu_temperature_c'],gpu['temperature_c'])
        super().enforce(sample,gpus)
DATA_SHA = '128bf3c84c140147fa8332d94488b67c1e123828e9b1135b761cb8f547d158e6'


def record(frequency):
    if type(frequency) is not int or frequency not in FREQUENCIES:
        raise ValueError('only the five requested frequencies are authorized')
    path = ROOT/'sources/Alu_1_2FBG_2PZT.mat'
    if sha256_file(path) != DATA_SHA:
        raise ValueError('experimental data bytes changed')
    rows = loadmat(path, simplify_cells=True)['Alu_1_2FBG_2PZT']
    matches = [r for r in rows if r['centerFreq'] == frequency]
    if len(matches) != 1:
        raise ValueError('expected one experimental record')
    row = matches[0]
    if row['numCycles'] != 5 or row['sampleRate'] != 5000000:
        raise ValueError('experimental excitation or sample rate changed')
    return row


def point_row(layout, xyz):
    """Exact tensor interpolation at a physical point, including either surface."""
    cells, p = layout.cells, layout.order
    lengths = (layout.plate.width_m, layout.plate.height_m, layout.plate.thickness_m)
    basis = _basis(line_rule(p)[0]); ids = []; values = []
    for x, length, count in zip(xyz, lengths, cells):
        if not 0 <= x <= length:
            raise ValueError('point outside plate')
        cell = min(int(math.floor(x/length*count)), count-1)
        local = 2*(x/length*count-cell)-1
        ids.append(cell*p+np.arange(p+1))
        values.append(np.array([b(local) for b in basis]))
    indices = ((ids[0][:, None, None]*(cells[1]*p+1)+ids[1][None, :, None])
               *(cells[2]*p+1)+ids[2][None, None, :]).ravel()
    weights = np.einsum('i,j,k->ijk', *values).ravel()
    keep = abs(weights) > 1e-18
    return indices[keep], weights[keep]


def combine(rows):
    ids = np.concatenate([r[0] for r in rows]); vals = np.concatenate([r[1] for r in rows])
    unique, inverse = np.unique(ids, return_inverse=True)
    weights = np.zeros(len(unique)); np.add.at(weights, inverse, vals)
    keep = abs(weights) > 1e-18
    return unique[keep], weights[keep]


def fbg_row(layout, z):
    # The 10 mm FBG is axial along x, centred 250 mm left of the actuator.
    left = point_row(layout, (.245, .5, z)); right = point_row(layout, (.255, .5, z))
    return combine([(right[0], right[1]/.01), (left[0], -left[1]/.01)])


def radial_edge_load(layout, quadrature=24):
    """Consistent ring traction: integral N (cos(theta),sin(theta),0) r dtheta.

    Integrate separately between crossings of the structured element edges.
    The load has zero resultant and torque; its xx and yy moments are pi*r^2
    for unit edge traction (1 N/m). This is an ideal bonded-wafer approximation.
    """
    radius = .005; cuts = [0., 2*math.pi]
    for length, count, axis in zip((1., 1.), layout.cells[:2], (0, 1)):
        for edge in np.linspace(0, length, count+1):
            ratio = (edge-.5)/radius
            if -1 < ratio < 1:
                if axis == 0:
                    angle = math.acos(ratio); cuts.extend((angle, 2*math.pi-angle))
                else:
                    angle = math.asin(ratio); cuts.extend((angle % (2*math.pi), (math.pi-angle) % (2*math.pi)))
    cuts = sorted(set(cuts)); nodes, weights = leggauss(quadrature)
    ns = layout.screen['scalar_dofs']; rows = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        for theta, weight in zip((nodes+1)*(b-a)/2+a, weights*(b-a)/2*radius):
            cosine, sine = math.cos(theta), math.sin(theta)
            ids, vals = point_row(layout, (.5+radius*cosine, .5+radius*sine, .001))
            rows.extend(((ids, vals*weight*cosine), (ids+ns, vals*weight*sine)))
    ids, vals = combine(rows); force = np.zeros(3*ns); force[ids] = vals
    return force


def bloch_frequency(k, cells, order, target_omega):
    """Independent 2D Bloch eigenproblem, with free top/bottom, for mesh screening."""
    q = order+1; hx = 1/cells; hz = .001
    _, w, d, *_ = line_rule(order)
    dx = np.kron(d, np.eye(q))*2/hx
    dz = np.kron(np.eye(q), d)*2/hz
    mass = np.kron(w, w)*hx*hz/4
    lam, mu = MATERIAL.lame
    xx = dx.T@(mass[:, None]*dx); zz = dz.T@(mass[:, None]*dz)
    xz = dx.T@(mass[:, None]*dz)
    stiffness = np.block([[(lam+2*mu)*xx+mu*zz, lam*xz+mu*xz.T],
                          [lam*xz.T+mu*xz, mu*xx+(lam+2*mu)*zz]])
    p = np.zeros((q*q, order*q), complex)
    p[:order*q] = np.eye(order*q)
    p[order*q:, :q] = np.eye(q)*np.exp(1j*k*hx)
    zero = np.zeros_like(p); projection = np.block([[p, zero], [zero, p]])
    mm = np.tile(mass*MATERIAL.density_kg_m3, 2)
    eigen, vectors = eigh(projection.conj().T@stiffness@projection,
                         projection.conj().T@(mm[:, None]*projection))
    full = (projection@vectors).reshape(2, q, q, -1)
    # Select A parity (ux odd and uz even through thickness), then the unfolded
    # branch near the specified frequency. The lowest Bloch eigenvalue aliases
    # at the Brillouin boundary and is not necessarily the requested branch.
    reflected = full[:, :, ::-1].copy(); reflected[0] *= -1
    parity = np.linalg.norm((full-reflected).reshape(2*q*q, -1), axis=0)
    parity /= np.linalg.norm(full.reshape(2*q*q, -1), axis=0)
    candidates = np.flatnonzero((parity < 1e-6) & (eigen > 0))
    if not len(candidates):
        raise ArithmeticError('Bloch A0 branch not positive')
    selected = candidates[np.argmin(abs(np.sqrt(eigen[candidates])-target_omega))]
    return math.sqrt(eigen[selected])


def mesh_screen(cells, order, frequency):
    mode = mode_velocity(frequency, .001, 'A0', MATERIAL)
    k = mode.wavenumber_rad_m; dk = k*1e-3
    target = 2*math.pi*frequency
    omega = bloch_frequency(k, cells, order, target)
    group = (bloch_frequency(k+dk, cells, order, target)-bloch_frequency(k-dk, cells, order, target))/(2*dk)
    return dict(frequency_hz=frequency, cells=cells, order=order,
                phase_relative_error=omega/(2*math.pi*frequency)-1,
                group_relative_error=group/mode.group_m_s-1,
                continuum_group_m_s=mode.group_m_s)


def arrival(time_s, signal, frequency):
    """Paper's theory-centred packet window; no fitted timing shift or time warp."""
    theory = .25/mode_velocity(frequency, .001, 'A0', MATERIAL).group_m_s
    pulse = 5/frequency
    # Use the same 350 us interval and zero padding for either signal's envelope.
    dt = .2e-6; t = np.arange(1750)*dt
    if time_s[0] > t[0] or time_s[-1] < t[-1]:
        raise ValueError('arrival extraction requires the complete common interval')
    y = np.interp(t, time_s, signal)
    y -= y[t < 20e-6].mean()
    envelope = abs(hilbert(y, 8192)[:len(y)])
    selected = np.flatnonzero((t >= theory) & (t <= theory+pulse))
    at = selected[np.argmax(envelope[selected])]
    return dict(packet_peak_s=float(t[at]), flight_s=float(t[at]-pulse/2),
                group_m_s=float(.25/(t[at]-pulse/2)),
                window_s=[theory, theory+pulse],
                peak_at_window_edge=bool(at in (selected[0], selected[-1])))


def source_hashes():
    paths = sorted((PROJECT/'traxis').glob('*.py'))
    paths.extend((Path(__file__), PROJECT/'examples/embedded_wave.py'))
    return {str(p.relative_to(PROJECT)).replace('\\', '/'):sha256_file(p) for p in paths}


def protocol():
    return dict(kind='five_experimental_plate_timing_comparisons',
        frequencies_hz=list(FREQUENCIES), plate=asdict(PLATE), material=asdict(MATERIAL),
        alloy='not specified in source; must not be labelled 6061-T6',
        dataset='https://zenodo.org/records/18955122',
        manuscript='https://zenodo.org/records/18937842',
        data_file='Alu_1_2FBG_2PZT.mat', data_sha256=DATA_SHA,
        source_center_m=[.5,.5,.001], source_radius_m=.005,
        source='ideal radial edge traction driven by recorded generator waveform; 1 N/m peak',
        receiver='10 mm x-directed FBG mean axial strain; paired top/bottom at x=.25,y=.5',
        comparison='direct A0 packet envelope peak timing from (top-bottom)/2',
        cells=list(CELLS), order=ORDER, steps=STEPS, duration_s=DURATION,
        output_stride=8, energy_stride=256, max_vector_dofs=32000000,
        max_device_mib=6144, max_host_mib=4096, max_workspace_mib=4096,
        limitations=['PZT/glue/fibre mass and stiffness omitted',
                    'voltage-to-traction and strain-to-voltage transfer uncalibrated',
                    'source does not specify boundary supports or dimensional tolerances',
                    'free boundary model; no damping or electronics',
                    'Bloch screening is not full spatial/time convergence',
                    'one plate at five frequencies, not five independent specimens',
                    'exploratory arrival timing, not acceptance for training data'],
        acceptance_threshold=None, source_sha256=source_hashes())


def worker(frequency, output):
    started = time.perf_counter(); output=Path(output); output.mkdir()
    expected = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    if expected != protocol():
        raise ValueError('frozen experimental protocol or source bytes changed')
    guard = PeakGuard(ResourceLimits(max_seconds=86400, max_rss_mib=4096))
    result = dict(completed=False, ready_for_training_data=False, frequency_hz=frequency)
    def phase(name, **extra):
        print(json.dumps(dict(phase=name, elapsed_seconds=time.perf_counter()-started, **extra)),flush=True)
    try:
        row=record(frequency); guard.check(force=True)
        layout=make_layout(PLATE,CELLS,ORDER,max_dofs=32000000,max_device_mib=6144,max_host_mib=4096)
        system=build_gpu_tensor(layout,MATERIAL,guard=guard)
        if DURATION/STEPS > .8*system.stable_dt_s:
            raise ValueError('time step exceeds frozen stability margin')
        phase('operator_ready',dofs=layout.screen['vector_dofs'],stable_dt_s=system.stable_dt_s)
        load=radial_edge_load(layout)
        top=fbg_row(layout,.001); bottom=fbg_row(layout,0.)
        probes=sparse_probe_set([top,bottom],layout.screen['vector_dofs'])
        drive=row['nifgen_avg_waveform_ch1']; scale=float(np.max(abs(drive)))
        drive_t=np.arange(len(drive))/row['sampleRate']; normalized_drive=drive/scale
        cooling=ThermalPause(guard,76.,72.); last=-math.inf
        phase('ports_ready',source_support=int(np.count_nonzero(load)))
        def progress(event):
            nonlocal last
            cooling.check(event['steps_completed'])
            if time.perf_counter()-last >= 15 or event['steps_completed']==STEPS:
                phase('propagating',**event);last=time.perf_counter()
                write_json(output/f"progress-{event['steps_completed']:06d}.json",
                           dict(**event,elapsed_seconds=last-started))
        simulation=gpu_explicit_simulate(system,STEPS,DURATION/STEPS,probes,load,
            force=lambda t:float(np.interp(t,drive_t,normalized_drive,left=0.,right=0.)),
            output_stride=8,energy_stride=256,guard=guard,progress=progress,max_workspace_mib=4096)
        phase('propagation_complete')
        gpu_force=system.stiffness@simulation.final_displacement_m
        # Release device pools before the independent, chunked CPU force check.
        system.stiffness.storage.cp.cuda.get_current_stream().synchronize()
        cpu_force=streamed_action(layout,MATERIAL,simulation.final_displacement_m,guard,chunk_elements=64)
        force_error=float(np.linalg.norm(cpu_force-gpu_force)/max(np.linalg.norm(cpu_force),1e-300))
        if force_error>1e-10:
            raise ArithmeticError('independent final-state CPU force check failed')
        traces=simulation.displacement_m
        experimental=row['niscope_avg_waveform']
        exp_t=np.arange(len(experimental))/row['sampleRate']
        observed=(experimental[:,0]-experimental[:,1])/2
        predicted=(traces[:,0]-traces[:,1])/2
        real=arrival(exp_t,observed,frequency)
        model=arrival(simulation.time_s,predicted,frequency)
        tail=simulation.energy_time_s>=5/frequency+2*DURATION/STEPS
        energies=simulation.modified_energy_j[tail]
        energy_drift=float(np.ptp(energies)/max(abs(energies[0]),1e-300))
        if energy_drift>1e-8:
            raise ArithmeticError('post-excitation modified energy drift failed')
        atomic_write(output/'waveforms.npz', lambda f:np.savez_compressed(f,
            simulation_time_s=simulation.time_s,top_axial_strain=traces[:,0],bottom_axial_strain=traces[:,1],
            experimental_time_s=exp_t,experimental_top_fbg_v=experimental[:,0],experimental_bottom_fbg_v=experimental[:,1],
            drive_time_s=drive_t,drive_v=drive,energy_time_s=simulation.energy_time_s,
            physical_energy_j=simulation.physical_energy_j,modified_energy_j=simulation.modified_energy_j))
        atomic_write(output/'final-state.npz',lambda f:np.savez_compressed(f,
            displacement_m=simulation.final_displacement_m,velocity_m_s=simulation.final_velocity_m_s))
        result.update(completed=True, experimental=real,simulation=model,
            arrival_error_us=(model['packet_peak_s']-real['packet_peak_s'])*1e6,
            flight_relative_error=model['flight_s']/real['flight_s']-1,
            cpu_force_relative_error=force_error,post_burst_modified_energy_relative_drift=energy_drift,
            cooling_seconds=cooling.pause_seconds,wall_seconds=time.perf_counter()-started,
            protocol_sha256=sha256_file(ROOT/'protocol.json'),source_sha256=source_hashes(),
            system=system.screen,device=system.device,statistics=simulation.statistics,
            waveforms_sha256=sha256_file(output/'waveforms.npz'),final_state_sha256=sha256_file(output/'final-state.npz'))
        phase('comparison_complete',arrival_error_us=result['arrival_error_us'],flight_relative_error=result['flight_relative_error'])
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}',wall_seconds=time.perf_counter()-started)
        raise
    finally:
        result['resource_samples']=guard.history
        result['resource_peaks']=getattr(guard,'peaks',{})
        write_json(output/'result.json',result)


def run_one(frequency):
    record(frequency)
    output=ROOT/f'trial-{FREQUENCIES.index(frequency)+1}-{frequency//1000}khz'
    if output.exists():
        raise FileExistsError('preserve all attempts; do not overwrite an existing trial')
    output.mkdir()
    command=[sys.executable,'-m','examples.experimental_five','worker',str(frequency),str(output/'native')]
    guard=PeakGuard(ResourceLimits(max_seconds=86500,max_rss_mib=4096))
    report=dict(completed=False,frequency_hz=frequency)
    try:
        environment={**os.environ,'CUPY_CACHE_DIR':str(ROOT/'cupy-cache')}
        report.update(run_monitored(command,PROJECT,environment,guard,output/'worker.log'))
        report['completed']=report['exit_code']==0
    finally:
        report['resource_samples']=guard.history
        report['resource_peaks']=getattr(guard,'peaks',{})
        write_json(output/'launcher.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('screen','freeze','worker','run'))
    parser.add_argument('frequency',type=int,nargs='?')
    parser.add_argument('output',nargs='?')
    args = parser.parse_args()
    if args.action=='screen':
        for f in (*FREQUENCIES,350000):
            print(json.dumps(mesh_screen(CELLS[0],ORDER,f)),flush=True)
    elif args.action=='freeze':
        path=ROOT/'protocol.json'
        if path.exists():raise FileExistsError('protocol already frozen')
        write_json(path,protocol())
        with zipfile.ZipFile(ROOT/'simulation-sources.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for name in source_hashes():archive.write(PROJECT/name,name)
            archive.write(path,'protocol.json')
        write_json(ROOT/'source-archive.json',dict(sha256=sha256_file(ROOT/'simulation-sources.zip'),
                   protocol_sha256=sha256_file(path)))
    elif args.action=='worker':worker(args.frequency,args.output)
    else:
        report=run_one(args.frequency)
        print(json.dumps({k:v for k,v in report.items() if k!='resource_samples'}),flush=True)


if __name__ == '__main__':
    main()
