"""Compare saved v6/v7 voltage magnitudes without fitting gain or rerunning CUDA."""
import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from traxis.storage import atomic_write, sha256_file, write_json
from examples.plate_outputs import relocated

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT/'outputs/result-analyses/resolved-pzt-strength-20261119'


def magnitudes(signal, time_s):
    centered = signal - np.mean(signal[time_s < 20e-6])
    return dict(peak_to_peak_v=float(np.ptp(signal)),
                baseline_corrected_peak_v=float(np.max(abs(centered))),
                baseline_corrected_rms_v=float(np.sqrt(np.mean(centered**2))))


def main():
    # Same full shared interval/grid as the saved timing and shape comparisons.
    time_s = np.arange(1750)*2e-7
    rows = []
    sources = []
    fig, axes = plt.subplots(5, 2, figsize=(12, 12), sharex=True)
    for index, khz in enumerate((50, 100, 150, 200, 250)):
        version = 'v6' if khz < 150 else 'v7'
        directory = relocated(PROJECT/f'outputs/wandowski-resolved-pzt-{version}/{khz}khz/native')
        path = directory/'waveforms.npz'
        result = json.loads((directory/'result.json').read_text())
        checksum = sha256_file(path)
        if not result['completed'] or checksum != result['waveforms_sha256']:
            raise ValueError(f'Incomplete or changed waveform: {path}')
        with np.load(path) as saved:
            pairs = []
            for prefix, time_key, voltage_key in (
                    ('simulation', 'simulation_time_s', 'ideal_open_reference_voltage_v'),
                    ('measured', 'experimental_time_s', 'experimental_pzt_voltage_v')):
                t, v = saved[time_key], saved[voltage_key]
                if (t[0] > time_s[0] or t[-1] < time_s[-1] or v.shape != (len(t), 2)
                        or not np.isfinite(v).all() or not np.all(np.diff(t) > 0)):
                    raise ValueError(f'Invalid {prefix} voltage/time data')
                pairs.append(np.column_stack([np.interp(time_s, t, v[:, j]) for j in range(2)]))
            sim, real = pairs
            drive_peak = float(np.max(abs(saved['recorded_generator_voltage_v'])))
            reference_peak = float(np.max(abs(saved['unit_peak_reference_voltage_v'])))
            if not np.isclose(reference_peak, 1.):
                raise ValueError('Saved simulation drive was not the expected unit peak')
        sources.append(dict(frequency_khz=khz, waveform=str(path.relative_to(PROJECT)),
                            waveform_sha256=checksum, result_sha256=sha256_file(directory/'result.json'),
                            stored_generator_waveform_peak=drive_peak,
                            simulated_actuator_peak_v=reference_peak))
        for channel, sim_trace, real_trace in (
                ('top', sim[:, 0], real[:, 0]), ('bottom', sim[:, 1], real[:, 1]),
                ('top_plus_bottom', sim[:, 0]+sim[:, 1], real[:, 0]+real[:, 1]),
                ('top_minus_bottom', sim[:, 0]-sim[:, 1], real[:, 0]-real[:, 1])):
            predicted = magnitudes(sim_trace, time_s)
            observed = magnitudes(real_trace, time_s)
            row = dict(frequency_khz=khz, channel=channel)
            for metric in predicted:
                row['simulation_'+metric] = predicted[metric]
                row['measured_'+metric] = observed[metric]
            row['measured_to_simulated_peak_to_peak_ratio'] = (
                observed['peak_to_peak_v']/predicted['peak_to_peak_v']
                if predicted['peak_to_peak_v'] else None)
            rows.append(row)
        for channel in range(2):
            ax = axes[index, channel]
            ax.plot(time_s*1e6, real[:, channel], lw=.8, label='Measured')
            ax.plot(time_s*1e6, sim[:, channel], lw=.8, label='Simulation: 1 V peak drive')
            ax.set_title(f'{khz} kHz | {("top", "bottom")[channel]} receiver')
            ax.set_ylabel('Voltage (V)'); ax.grid(alpha=.2)
            if index == 0:ax.legend(fontsize=8)
            if index == 4:ax.set_xlabel('Time (microseconds)')
    fig.suptitle('Raw receiver voltages: no amplitude normalization or fitted gain\n'
                 'Actuator amplifier gain and receiver loading are uncalibrated', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure = OUTPUT/'raw-voltage-comparison.png'
    if figure.exists():raise FileExistsError(figure)
    fig.savefig(figure, dpi=140);plt.close(fig)
    evidence = dict(
        method='Unfitted voltages, common 5 MHz grid from 0 through 349.8 microseconds. '
               'Peak-to-peak=max-min. RMS/absolute peak subtract mean of first 20 microseconds. '
               'No gain, polarity, amplitude normalization or time shift fitted.',
        absolute_strength_validated=False,
        limitations=[
            'Simulation applies a unit-peak actuator voltage; actual amplified actuator voltage is unknown.',
            'Generator waveform alone does not establish voltage at PZT after the amplifier.',
            'Ideal open-circuit simulated receivers omit actual load/cable transfer and bond losses.',
            'Raw magnitude ratios are not calibrated physical amplitude-error percentages.',
            'Whole-interval magnitudes can include multiple modes and reflections; sums/differences are not verified pure modes.'],
        analysis_sha256=sha256_file(Path(__file__)), sources=sources, records=rows,
        figure_sha256=sha256_file(figure))
    write_json(OUTPUT/'comparison.json', evidence)
    text = io.StringIO(); writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
    atomic_write(OUTPUT/'comparison.csv', lambda stream: stream.write(text.getvalue().encode()))
    for row in rows:
        if row['channel'] in ('top', 'bottom'):
            print(f"{row['frequency_khz']} kHz {row['channel']}: "
                  f"simulated {row['simulation_peak_to_peak_v']:.6f} Vpp; "
                  f"measured {row['measured_peak_to_peak_v']:.6f} Vpp; "
                  f"ratio {row['measured_to_simulated_peak_to_peak_ratio']:.2f}")
    print('Saved:', OUTPUT)


if __name__ == '__main__':main()
