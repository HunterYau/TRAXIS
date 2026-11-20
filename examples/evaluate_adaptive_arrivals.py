"""Compare timing methods on saved recordings; never launches a simulation."""
import json
from pathlib import Path
import numpy as np
from examples.plate_outputs import PROJECT, RUNS_ROOT, result_directory, text_file
from examples.resolved_pzt_queue import compare
from traxis.storage import sha256_file,write_json


def main():
    output=PROJECT/'outputs/result-analyses/adaptive-arrival-window-v2'
    output.mkdir(parents=True,exist_ok=False)
    rows=[];lines=['Saved-record comparison: fixed versus adaptive arrival windows',
        'Lower timing error does not prove that a packet is correctly identified.',
        'A0 above the published detection range remains tentative.', '']
    for kind in ('ideal','realistic'):
        for khz in (50,100,150,200,250):
            name=('run-006-wandowski-ideal-bond-50-100khz' if khz<150 else
                  'run-007-wandowski-ideal-bond-150-250khz-continuation') if kind=='ideal' else 'run-009-wandowski-realistic-bond-damping'
            root=RUNS_ROOT/name;folder=result_directory(root/f'{khz}khz')
            result=json.loads((folder/'result.json').read_text())
            path=folder/'waveforms.npz'
            if not result['completed'] or sha256_file(path)!=result['waveforms_sha256']:
                raise ValueError('Incomplete or changed waveform: '+str(path))
            config=json.loads((root/'prepared.json').read_text())['configuration']
            with np.load(path) as data:
                voltage=data['ideal_open_reference_voltage_v'] if kind=='ideal' else data['simulated_receiver_voltage_v']
                comparison=compare(data['simulation_time_s'],voltage,data['experimental_pzt_voltage_v'],
                                   khz*1000,5000000,config=config)
            rows.append(dict(run=kind,frequency_khz=khz,waveforms_sha256=sha256_file(path),comparison=comparison))
            for label,mode in (('fast','S0'),('slow','A0')):
                fixed=comparison[label+'_fixed_window'];adaptive=comparison[label+'_adaptive_window']
                details=comparison[label+'_details']
                lines.append(f'{kind} {khz} kHz {mode}: fixed candidate {fixed["raw_candidate_error_percent"]:.2f}% '
                    f'(window valid: {fixed["reliable_window"]}); selected candidate {details["raw_candidate_error_percent"]:.2f}% '
                    f'| measured {adaptive["measured"]["status"]}, simulated {adaptive["simulated"]["status"]} '
                    f'| selected {comparison[label+"_arrival_method"]} | reported status {comparison[label+"_status"]}')
    write_json(output/'comparison.json',dict(records=rows,simulation_started=False))
    text_file(output/'summary.txt','\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
