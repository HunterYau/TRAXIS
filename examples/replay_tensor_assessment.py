"""Reproduce a tensor assessment from its retained, hash-checked source snapshot."""
import argparse
import json
import os
from pathlib import Path
import stat
import sys
import zipfile

from traxis.elmer_reference import run_monitored
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import canonical_hash,sha256_file,write_json


def checked_json(path):
    path=Path(path)
    if path.stat().st_size>4*1024**2:raise MemoryError('replay metadata exceeds 4 MiB')
    value=json.loads(path.read_text(encoding='utf-8'))
    if value.get('fingerprint')!=canonical_hash({k:v for k,v in value.items() if k!='fingerprint'}):
        raise ValueError('replay metadata fingerprint changed')
    return value


def extract_sources(study_path,output):
    study_path,output=Path(study_path).resolve(),Path(output).resolve();study=checked_json(study_path)
    if study['kind']!='frozen_tensor_wave_study':raise ValueError('replay needs a frozen tensor study')
    snapshot=study_path.parent/'source-snapshot.zip'
    if sha256_file(snapshot)!=study['source_snapshot_sha256']:raise ValueError('frozen source archive changed')
    expected={**{'traxis/'+n:h for n,h in study['source_sha256'].items()},
        **{'examples/'+n:h for n,h in study['driver_sha256'].items()}}
    if output.exists():raise FileExistsError('source replay needs a fresh extraction directory')
    output.mkdir(parents=True)
    with zipfile.ZipFile(snapshot) as archive:
        entries=archive.infolist()
        if (len(entries)!=len(expected)+1 or {e.filename for e in entries}!=set(expected)|{'case.json'}
                or sum(e.file_size for e in entries)>8*1024**2 or any(e.file_size>1024**2 for e in entries)):
            raise ValueError('frozen source archive has an unexpected inventory or size')
        for entry in entries:
            path=(output/entry.filename).resolve()
            if (not path.is_relative_to(output) or entry.is_dir()
                    or stat.S_ISLNK(entry.external_attr>>16)):
                raise ValueError('frozen source entry escapes its extraction directory')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:stream.write(archive.read(entry))
            if entry.filename in expected and sha256_file(path)!=expected[entry.filename]:
                raise ValueError('extracted numerical source disagrees with the frozen identity')
    return study,expected


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--mode',choices=('spatial','time'),required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();project=Path(__file__).resolve().parents[1];output=args.output.resolve()
    if not output.is_relative_to(project/'outputs') or output.exists():raise ValueError('replay needs a new project output directory')
    guard=ResourceGuard(ResourceLimits(max_seconds=180,max_rss_mib=1536,max_system_memory_percent=85,min_available_mib=2048))
    output.mkdir(parents=True);result=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        mode=args.mode,replay_driver_sha256=sha256_file(__file__),study_sha256=sha256_file(args.study))
    try:
        source=output/'frozen-source';study,expected=extract_sources(args.study,source)
        temporary=output/'temporary';temporary.mkdir()
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
            **{n:str(temporary) for n in ('TEMP','TMP','TMPDIR')}}
        command=[sys.executable,'-m','examples.tensor_wave_study',args.mode,'--plan',str(args.study.resolve()),'--output',str(output/'assessment')]
        result['worker_run']=run_monitored(command,source,env,guard,output/'worker.log')
        if result['worker_run']['exit_code'] not in (0,2):raise RuntimeError('frozen assessment replay failed; inspect worker.log')
        assessment=checked_json(output/'assessment'/f'{args.mode}-refinement-report.json')
        if (not assessment['completed'] or assessment['study_fingerprint']!=study['fingerprint']
                or any(sha256_file(source/n)!=h for n,h in expected.items())):
            raise ValueError('frozen assessment replay source/result changed')
        result.update(completed=True,passed=assessment['passed'],study_fingerprint=study['fingerprint'],
            frozen_source_snapshot_sha256=study['source_snapshot_sha256'],extracted_source_sha256=expected,
            assessment_fingerprint=assessment['fingerprint'],assessment_sha256=sha256_file(output/'assessment'/f'{args.mode}-refinement-report.json'),
            comparison_sha256=assessment['comparison_sha256'],receivers=assessment['receivers'],
            scope='Postprocessing of existing raw states using the exact retained numerical sources; no forward rerun, source override or data acceptance.')
    except Exception as error:result['error']=f'{type(error).__name__}: {error}'
    finally:
        result['resources']=guard.report();result['fingerprint']=canonical_hash(result);write_json(output/'replay-result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('resources','extracted_source_sha256','receivers')}))
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
