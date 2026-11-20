"""Prepared PZT plate queue. Default is preflight only; --run starts serial GPU trials.

No scheduler, import, or preflight starts a time integration. Unknown hardware
loads remain unknown; ideal-reference voltages are explicitly labelled.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from examples.experimental_five import PROJECT, PeakGuard, record
from examples.embedded_wave import ThermalPause
from traxis.config import Material, Plate
from traxis.gpu_tensor import make_layout, build_gpu_tensor, streamed_action
from traxis.gpu_enriched_verlet import gpu_explicit_simulate, sparse_probe_set
from traxis.piezoelectric import PiezoMaterial, Readout, read_profile
from traxis.pzt_receiver import DiskGeometry, disk_receiver
from traxis.resources import ResourceLimits
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json
from traxis.elmer_reference import run_monitored
from examples.plate_outputs import relocated
from examples.output_locations import organized_output

CONFIG = PROJECT/'configs/pzt/wandowski-2025-plate1.json'
ROOT = relocated(PROJECT/'outputs/wandowski-pzt-plate1')


def configuration():
    data = json.loads(CONFIG.read_text(encoding='utf-8'))
    if (data['frequencies_hz'] != [50000, 100000, 150000, 200000, 250000]
            or data['auto_start'] or data['ready_for_training_data']):
        raise ValueError('unexpected queue or acceptance state')
    return data


def identities():
    files = [*sorted((PROJECT/'traxis').glob('*.py')), Path(__file__), CONFIG,
             PROJECT/'examples/experimental_five.py', PROJECT/'examples/embedded_wave.py',
             PROJECT/configuration()['material_profile']]
    return {str(p.relative_to(PROJECT)).replace('\\', '/'): sha256_file(p) for p in files}


def measured_channels(row):
    """Dataset README order: FBG top, FBG bottom, PZT top, PZT bottom."""
    signals = np.asarray(row['niscope_avg_waveform'])
    if signals.ndim != 2 or signals.shape[1] != 4 or not np.isfinite(signals).all():
        raise ValueError('invalid measured channels')
    return signals[:, [2, 3]]


def preflight():
    config = configuration()
    if sha256_file(organized_output(PROJECT/config['data_file'])) != config['data_sha256']:
        raise ValueError('reference dataset checksum mismatch')
    material = PiezoMaterial.from_profile(read_profile(PROJECT/config['material_profile']))
    cases = []
    for frequency in config['frequencies_hz']:
        row = record(frequency)
        if row['Number_of_averages'] != config['averages']:
            raise ValueError('unexpected averaging')
        measured = measured_channels(row)
        drive = np.asarray(row['nifgen_avg_waveform_ch1'])
        if drive.ndim != 1 or not np.isfinite(drive).all() or np.max(abs(drive)) <= 0:
            raise ValueError('invalid recorded excitation')
        if np.any(np.ptp(measured, axis=0) == 0):
            raise ValueError('empty PZT measurement')
        cases.append(dict(frequency_hz=frequency, measured_samples=len(measured),
                          measured_channels=[2, 3], status='prepared_not_run'))
    result = dict(configuration=config, cases=cases, source_sha256=identities(),
                  planar_coupling_c_m2=material.planar().coupling_c_m2.tolist(),
                  simulation_started=False, ready_for_training_data=False)
    result['fingerprint'] = canonical_hash(result)
    return result


def ports(layout, config, guard=None):
    planar = PiezoMaterial.from_profile(read_profile(PROJECT/config['material_profile'])).planar()
    def disk(center, surface):
        return disk_receiver(layout, planar, DiskGeometry(tuple(center), config['disk_radius_m'],
            config['disk_thickness_m'], surface, 1), guard=guard)
    source = disk(config['actuator_center_xy_m'], 'top')
    receivers = [disk(config['receiver_center_xy_m'], surface) for surface in ('top', 'bottom')]
    # Q_top = C V - q_strain. Mechanical internal force is K u + g V;
    # a prescribed voltage therefore supplies external force -g V.
    force = np.zeros(layout.screen['vector_dofs'])
    force[source.indices] = -source.charge_weights_c_m
    probes = sparse_probe_set([(r.indices, r.charge_weights_c_m) for r in receivers], len(force))
    return force, probes, receivers


def worker(frequency, directory):
    frozen = json.loads((ROOT/'prepared.json').read_text())
    if frozen != preflight():
        raise ValueError('prepared source/config/data identity changed; preserve queue and prepare a new version')
    config = frozen['configuration']
    if frequency not in config['frequencies_hz']:
        raise ValueError('frequency not prepared')
    directory = Path(directory); directory.mkdir(exist_ok=False)
    guard = PeakGuard(ResourceLimits(max_seconds=86400, max_rss_mib=4096))
    started = time.perf_counter()
    result = dict(completed=False, frequency_hz=frequency, ready_for_training_data=False,
                  protocol_fingerprint=frozen['fingerprint'])
    def event(phase, **details):
        print(json.dumps(dict(phase=phase, elapsed_seconds=time.perf_counter()-started, **details)), flush=True)
    try:
        guard.check(force=True)
        mesh = config['mesh']; row = record(frequency)
        layout = make_layout(Plate(*config['plate_m']), tuple(mesh['cells']), mesh['order'],
            max_dofs=32000000, max_device_mib=6144, max_host_mib=4096)
        system = build_gpu_tensor(layout, Material(**config['host_material']), guard=guard)
        dt = mesh['duration_s']/mesh['steps']
        if dt > .8*system.stable_dt_s:
            raise ValueError('time step exceeds stability margin')
        force, probes, receivers = ports(layout, config, guard)
        event('ports_ready', receiver_reports=[r.report for r in receivers])
        drive = np.asarray(row['nifgen_avg_waveform_ch1'])
        drive_t = np.arange(len(drive))/row['sampleRate']; reference = drive/np.max(abs(drive))
        cooling = ThermalPause(guard, 76., 72.); last = -math.inf
        def progress(state):
            nonlocal last
            cooling.check(state['steps_completed'])
            if time.perf_counter()-last >= 15 or state['steps_completed'] == mesh['steps']:
                event('propagating', **state); last = time.perf_counter()
        simulation = gpu_explicit_simulate(system, mesh['steps'], dt, probes, force,
            force=lambda t: float(np.interp(t, drive_t, reference, left=0., right=0.)),
            output_stride=mesh['output_stride'], energy_stride=256, guard=guard,
            progress=progress, max_workspace_mib=4096)
        # CUDA sparse-probe output is CHARGE, despite the generic sampler field name.
        charge = simulation.displacement_m
        voltage = np.column_stack([r.record(simulation.time_s, charge[:, j], Readout(None, 0.))['voltage_v']
                                   for j, r in enumerate(receivers)])
        measured = measured_channels(row)
        atomic_write(directory/'waveforms.npz', lambda f: np.savez_compressed(f,
            simulation_time_s=simulation.time_s, strain_charge_c=charge,
            ideal_open_reference_voltage_v=voltage,
            experimental_time_s=np.arange(len(measured))/row['sampleRate'],
            experimental_pzt_voltage_v=measured, drive_time_s=drive_t,
            recorded_generator_voltage_v=drive, unit_peak_reference_voltage_v=reference,
            energy_time_s=simulation.energy_time_s, modified_energy_j=simulation.modified_energy_j))
        gpu_force = system.stiffness@simulation.final_displacement_m
        system.stiffness.storage.cp.cuda.get_current_stream().synchronize()
        cpu_force = streamed_action(layout, Material(**config['host_material']),
            simulation.final_displacement_m, guard, chunk_elements=64)
        error = float(np.linalg.norm(cpu_force-gpu_force)/max(np.linalg.norm(cpu_force), 1e-300))
        tail = simulation.modified_energy_j[simulation.energy_time_s >= 5/frequency+2*dt]
        drift = float(np.ptp(tail)/max(abs(tail[0]), 1e-300))
        if error > 1e-10 or drift > 1e-8:
            raise ArithmeticError('independent force or post-burst energy check failed')
        atomic_write(directory/'final-state.npz', lambda f: np.savez_compressed(f,
            displacement_m=simulation.final_displacement_m, velocity_m_s=simulation.final_velocity_m_s))
        result.update(completed=True, cpu_force_relative_error=error,
            modified_energy_relative_drift=drift, receiver_reports=[r.report for r in receivers],
            waveforms_sha256=sha256_file(directory/'waveforms.npz'),
            final_state_sha256=sha256_file(directory/'final-state.npz'))
        event('trial_saved', frequency_hz=frequency, **{k:result[k] for k in ('cpu_force_relative_error','modified_energy_relative_drift')})
    finally:
        result.update(wall_seconds=time.perf_counter()-started, resource_peaks=getattr(guard, 'peaks', {}))
        write_json(directory/'result.json', result)


def run_queue():
    if json.loads((ROOT/'prepared.json').read_text()) != preflight():
        raise ValueError('prepared inputs changed')
    # Exclusive persistent ownership prevents concurrent launch and accidental restart.
    write_json(ROOT/'queue-owner.json', dict(pid=os.getpid(), started_unix=time.time()))
    for frequency in configuration()['frequencies_hz']:
        if (ROOT/'STOP_AFTER_CURRENT').exists():
            print('Stopped between trials; completed results retained.', flush=True); return
        trial = ROOT/f'{frequency//1000}khz'; trial.mkdir(exist_ok=False)
        guard = PeakGuard(ResourceLimits(max_seconds=86500, max_rss_mib=4096))
        command = [sys.executable, '-m', 'examples.wandowski_pzt_queue', '--worker', str(frequency), str(trial/'native')]
        report = run_monitored(command, PROJECT,
            {**os.environ, 'CUPY_CACHE_DIR':str(ROOT/'cupy-cache')}, guard, trial/'worker.log')
        write_json(trial/'launcher.json', report)
        if report['exit_code'] != 0:
            raise RuntimeError(f'{frequency} Hz failed; queue stopped and partial output preserved')
        print(json.dumps(dict(phase='trial_complete', frequency_hz=frequency, results=str(trial))), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--worker', nargs=2, metavar=('FREQUENCY_HZ', 'OUTPUT'))
    args = parser.parse_args()
    if args.worker:
        worker(int(args.worker[0]), args.worker[1]); return
    if args.run:
        run_queue(); return
    checked = preflight()
    if args.prepare:
        ROOT.mkdir(parents=True, exist_ok=True)
        write_json(ROOT/'prepared.json', checked)
    print(json.dumps(checked, indent=2))


if __name__ == '__main__':
    main()
