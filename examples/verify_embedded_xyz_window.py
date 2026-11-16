"""Run the unchanged independent CPU check for a coupled-XYZ control."""
import argparse
import json
from pathlib import Path

from examples.embedded_wave import checked_case
from examples.tensor_wave import PROJECT,read_json,sealed
from examples.verify_embedded_window import verify
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import sha256_file,write_json


def cpu_limits():
    return ResourceLimits(max_seconds=600,max_rss_mib=4096,
        max_system_memory_percent=89.9,min_available_mib=768)


def verify_xyz(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if not all(p.is_relative_to(PROJECT/'outputs') for p in (root,output)):
        raise ValueError('XYZ CPU inputs and outputs must remain in project outputs')
    case=checked_case(read_json(root/'embedded-wave-plan.json')['case'])
    if case['scope']!='xyz_refinement':
        raise ValueError('XYZ CPU launcher requires a declared coupled-XYZ case')
    if output.exists():
        raise FileExistsError('XYZ CPU output must be fresh')
    guard=ResourceGuard(cpu_limits());source_hash=sha256_file(__file__)
    report=None;finished=False
    try:
        report=verify(root,output,guard)
        if sha256_file(__file__)!=source_hash:
            raise ValueError('XYZ CPU launcher changed during checking')
        finished=True
        return report
    finally:
        proof=output/'cpu-window-check.json'
        if proof.exists():
            write_json(output/'cpu-launcher.json',sealed(dict(kind='embedded_xyz_cpu_launcher',
                completed=finished,passed=bool(finished and report['completed'] and report['passed']),
                ready_for_training_data=False,training_updates=0,launcher_sha256=source_hash,
                cpu_proof_sha256=sha256_file(proof),limits=guard.report()['limits'],
                note='Resource envelope only; numerical checking uses verify_embedded_window unchanged.')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=verify_xyz(args.run,args.output)
    print(json.dumps({k:report[k] for k in ('completed','passed','fingerprint')}))
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
