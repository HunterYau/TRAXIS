"""Verify the remedy snapshot and optional tiny GPU arithmetic. No plate solve."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))


def verify(*, gpu=False):
    from traxis.storage import sha256_file
    from examples import pzt_mor_remedies as study
    manifest = json.loads((PROJECT / 'remedies-transfer-manifest.json').read_text())
    for row in manifest['files']:
        path = (PROJECT / row['path']).resolve(); path.relative_to(PROJECT)
        if sha256_file(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Transfer mismatch: ' + row['path'])
    plan = study.load_plan(PROJECT / manifest['study_root'])
    if plan['preflight']['fingerprint'] != manifest['preflight_fingerprint']:
        raise ValueError('Scientific input fingerprint differs')
    if gpu:
        import cupy as cp
        import numpy as np
        from examples.pzt_mechanical_mor import project, impulse
        from examples.pzt_era_delay import fit_delayed, delayed_impulse
        m = cp.asarray([[2., .3], [.3, 1.]])
        k = cp.asarray([[300., -20.], [-20., 200.]])
        inv = cp.asarray(np.linalg.inv(cp.asnumpy(m)))
        c = cp.asarray([[1., .2], [.8, -.1]])
        model, _ = project(2, [1., .3], lambda x:c@x, lambda x:m@x,
                            lambda x:inv@x, lambda x:k@x, 2, xp=cp)
        actual, _ = impulse(model, 2, 65, .001)
        reference, _ = project(2, [1., .3], lambda x:cp.asnumpy(c)@x,
            lambda x:cp.asnumpy(m)@x, lambda x:cp.asnumpy(inv)@x, lambda x:cp.asnumpy(k)@x, 2)
        expected, _ = impulse(reference, 2, 65, .001)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        t=np.arange(65)*.001; h=np.zeros((65,2)); h[10:]=.97**np.arange(55)[:,None]*[[1.,.8]]
        model,_=fit_delayed(t,h,1,backend='cupy',delay_samples=8)
        actual,_=delayed_impulse(model,1,len(t))
        np.testing.assert_allclose(actual,h,atol=1e-11)
        cp.cuda.Device().synchronize()
        print('PASS tiny FP64 GPU BLAS, mechanical projection, FFT delay ERA; no full plate calculation.')
    print(f'PASS {len(manifest["files"])} hashes and frozen remedy plan. No schedule started.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--gpu',action='store_true')
    verify(gpu=parser.parse_args().gpu)
