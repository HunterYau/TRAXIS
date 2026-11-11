"""Retain a bounded 3.175 mm mounting sensitivity control without training."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import zipfile

import numpy as np

from traxis.config import Material, Plate
from traxis.forward import ForwardSettings, generate_waveform_bank
from traxis.mounting import PZTMounting
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.scenarios import AcquisitionSettings, NuisanceSettings, Scenario
from traxis.storage import canonical_hash, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]; output = args.output.resolve()
    if not output.is_relative_to(project/'outputs'):
        raise ValueError('mounting reference must remain in project outputs')
    output.mkdir(parents=True, exist_ok=False)
    guard = ResourceGuard(ResourceLimits(max_seconds=90, max_rss_mib=768,
        max_system_memory_percent=85, min_available_mib=2048)); guard.check(force=True)
    source_files = [*sorted((project/'traxis').glob('*.py')), Path(__file__)]
    sources = {str(p.relative_to(project)).replace('\\', '/'):sha256_file(p) for p in source_files}
    with zipfile.ZipFile(output/'source-snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(project/name, name)
    report = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0,
        source_sha256=sources, source_snapshot_sha256=sha256_file(output/'source-snapshot.zip'))
    try:
        nominal = Scenario('mount-debug', 'mount-debug', 'validation', 171, 43,
            Plate(.04, .04, .003175), (),
            AcquisitionSettings(inset_m=.006, patch_radius_m=.003, frequencies_hz=(100000.,), samples=64, cycles=2),
            NuisanceSettings(Material(), 1., (1.,)*8, (1.,)*8, relative_noise_std=0.))
        offsets = np.zeros((8, 2)); offsets[0] = [.0004, -.0003]
        mounting = PZTMounting(tuple(map(tuple, offsets)), (1.1,)+(1.,)*7)
        cases = dict(nominal=nominal,
            gain=replace(nominal, nuisance=replace(nominal.nuisance, sensor_gains=(1.1,)+(1.,)*7)),
            moved_contact=replace(nominal, mounting=mounting),
            remounted_current=replace(nominal, current_mounting=mounting))
        banks, rows = {}, {}
        settings = ForwardSettings(mesh_backend='structured')
        for name, case in cases.items():
            bank, receipt = generate_waveform_bank(case, settings, cache_dir=output/'cache', guard=guard)
            banks[name] = bank
            rows[name] = dict(scenario=case.to_dict(), forward_solves=receipt['forward_solves'], cache_hits=receipt['cache_hits'],
                internal_steps=receipt['internal_steps'], nominal_positions_m=bank.sensors_xy_m.tolist(),
                actual_intact_positions_m=case.sensor_geometry('intact')[0].tolist(),
                actual_current_positions_m=case.sensor_geometry('current')[0].tolist())
            np.testing.assert_array_equal(bank.sensors_xy_m, nominal.acquisition.sensors(nominal.plate))
            assert not bank.depth_mm.any() and not receipt['ready_for_training_data']
        base = banks['nominal']; moved = banks['moved_contact']; remount = banks['remounted_current']
        ratio = lambda a,b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
        shift_effect = ratio(moved.waveforms, base.waveforms)
        remount_residual = ratio(remount.waveforms, remount.baseline)
        factors = np.array([1.1 if a.transmitter == 0 or a.receiver == 0 else 1. for a in base.actions])
        gain_error = ratio(banks['gain'].waveforms, base.waveforms*factors[:, None])
        if (min(shift_effect, remount_residual) <= 1e-6 or gain_error > 1e-6
                or rows['gain']['forward_solves'] != 0 or rows['gain']['cache_hits'] != 7
                or rows['moved_contact']['forward_solves'] != 7 or rows['remounted_current']['forward_solves'] != 7):
            raise ArithmeticError('mounting sensitivity or cache behavior check failed')
        np.savez_compressed(output/'mounting-proofs.npz', time_s=base.time_s,
            nominal_waveforms=base.waveforms, moved_contact_waveforms=moved.waveforms,
            gain_waveforms=banks['gain'].waveforms, expected_gain_factors=factors,
            remount_current=remount.waveforms, remount_baseline=remount.baseline)
        guard.check(force=True)
        report.update(completed=True, passed=True, cases=rows,
            moved_contact_relative_change=shift_effect, intact_remount_relative_residual=remount_residual,
            gain_scaling_relative_error=gain_error, proofs_sha256=sha256_file(output/'mounting-proofs.npz'),
            scope='Small mechanical sensitivity and cache control; no calibrated clamp model, wave convergence or research-bank acceptance.')
        if any(sha256_file(project/name) != digest for name, digest in sources.items()):
            raise ValueError('source changed during mounting reference')
    except Exception as error:
        report.update(completed=False, passed=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['resources'] = guard.report(); report['fingerprint'] = canonical_hash(report)
        write_json(output/'mounting-report.json', report)
    print(json.dumps({k:report[k] for k in ('completed', 'passed', 'moved_contact_relative_change',
        'intact_remount_relative_residual', 'gain_scaling_relative_error', 'fingerprint')}))


if __name__ == '__main__':
    main()
