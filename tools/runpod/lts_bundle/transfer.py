"""Explicit SSH upload/download helper. No Runpod API or pod lifecycle calls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[3]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('action',choices=['upload','download'])
    p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--key',type=Path,required=True);p.add_argument('--known-hosts',type=Path,required=True)
    p.add_argument('--destination',type=Path,help='New local folder for a download')
    p.add_argument('--remote-root',default='/workspace/traxis-lts-six-frequency-003')
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*',args.host) or not 1<=args.port<=65535:
        raise ValueError('Invalid SSH host/port')
    if not re.fullmatch(r'/workspace/traxis-lts-[A-Za-z0-9_-]+',args.remote_root):
        raise ValueError('Use a new /workspace/traxis-lts-NAME folder')
    key=args.key.resolve(strict=True);known=args.known_hosts.resolve(strict=True)
    options=['-i',str(key),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
        '-o','UserKnownHostsFile='+str(known)]
    host='root@'+args.host
    def ssh(command):
        return subprocess.check_output(['ssh',*options,'-p',str(args.port),host,command],text=True).strip()
    def scp(source,target):
        subprocess.run(['scp',*options,'-P',str(args.port),str(source),str(target)],check=True)
    remote=args.remote_root;project=remote+'/TRAXIS'
    if args.action=='upload':
        bundle=ROOT/'outputs/runpod-transfer/lts-six-frequency-003'
        archive=bundle/'traxis-lts-six-frequency-003.tar.gz'
        expected=(bundle/'archive.sha256').read_text().split()[0]
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected: raise ValueError('Local archive checksum mismatch')
        # Atomic mkdir refuses to overwrite ANY existing pod folder.
        ssh('mkdir -- '+shlex.quote(remote))
        scp(archive,host+':'+remote+'/bundle.tar.gz')
        code='import hashlib; from pathlib import Path; assert hashlib.sha256(Path('+repr(remote+'/bundle.tar.gz')+').read_bytes()).hexdigest()=='+repr(expected)
        ssh('python3 -c '+shlex.quote(code)+' && tar -xzf '+shlex.quote(remote+'/bundle.tar.gz')+' -C '+shlex.quote(remote))
        print('Uploaded into '+project+'. Run setup.sh separately; no simulation started.')
    else:
        info=json.loads(ssh('python3 '+shlex.quote(project+'/tools/runpod/lts_bundle/export_results.py')))
        archive=info['archive']
        if not re.fullmatch(re.escape(project)+r'/lts-results-[0-9a-f]{12}\.tar\.gz',archive):
            raise ValueError('Unexpected exported archive path')
        dest=args.destination or ROOT/'outputs/runpod-results'/('lts-six-frequency-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        dest.mkdir(parents=True,exist_ok=False)
        local=dest/Path(archive).name;scp(host+':'+archive,local)
        if hashlib.sha256(local.read_bytes()).hexdigest()!=info['sha256']: raise ValueError('Downloaded archive hash mismatch')
        with tarfile.open(local) as tf:
            for member in tf.getmembers():
                (dest/member.name).resolve().relative_to(dest.resolve())
                if not (member.isfile() or member.isdir()): raise ValueError('Link/special archive member rejected')
            tf.extractall(dest,filter='data')
        manifest=json.loads((dest/'results-sha256.json').read_text())
        for name,expected in manifest.items():
            result=dest/'lts-005'/name;result.resolve().relative_to((dest/'lts-005').resolve())
            if hashlib.sha256(result.read_bytes()).hexdigest()!=expected: raise ValueError('Downloaded result hash mismatch: '+name)
        print('Downloaded and verified '+str(len(manifest))+' result files into '+str(dest))


if __name__=='__main__': main()
