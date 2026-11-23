"""Check an extracted transfer snapshot without starting simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--gpu', action='store_true', help='also check FP64 CUDA operation; no wave simulation')
    args = parser.parse_args()
    manifest = json.loads((PROJECT / 'transfer-manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path = (PROJECT / row['path']).resolve()
        path.relative_to(PROJECT)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Transfer mismatch: ' + row['path'])
    from examples import resolved_pzt_queue as queue
    proof = queue.preflight()
    if proof['fingerprint'] != manifest['preflight_fingerprint']:
        raise ValueError('Scientific input fingerprint differs from the transfer snapshot')
    from examples.experimental_five import record
    from examples.pzt_analytic_source import analytic_source
    import numpy as np
    for frequency in proof['configuration']['frequencies_hz']:
        row = record(frequency)
        drive = np.asarray(row['nifgen_avg_waveform_ch1'])
        analytic_source(frequency, row['numCycles'], np.arange(len(drive)) / row['sampleRate'], drive)
    if (PROJECT / 'examples/pzt_broadband_study.py').exists():
        from examples.pzt_broadband_study import load_plan
        for name in ('runpod-broadband-benchmark', 'runpod-broadband-convergence'):
            load_plan(PROJECT / 'outputs/simulation-runs' / name)
        print('PASS: both unstarted broadband/direct schedules match the transferred scientific sources.')
    if (PROJECT / 'examples/pzt_mor_study.py').exists():
        from examples import pzt_mor_study as mor
        mor.load_plan(PROJECT / mor.STUDY)
        print('PASS: unstarted MOR schedule and all saved full-order reference checksums.')
    print(f'PASS: {len(manifest["files"])} file checksums, scientific preflight, and five measured input records.')
    if args.gpu:
        import cupy as cp
        kernel = cp.RawKernel('extern "C" __global__ void square(const double* x, double* y) '
                              '{ int i=threadIdx.x; y[i]=x[i]*x[i]; }', 'square')
        x = cp.asarray([1., 2., 3., 4.], dtype=cp.float64)
        y = cp.empty_like(x)
        kernel((1,), (4,), (x, y))
        cp.cuda.Device().synchronize()
        np.testing.assert_array_equal(cp.asnumpy(y), [1., 4., 9., 16.])
        print('PASS: FP64 CUDA kernel; CuPy', cp.__version__)
        if (PROJECT / 'examples/pzt_mor_study.py').exists():
            # Tiny library smoke check, not a plate simulation or a ROM build.
            from examples.pzt_era_cuda import exact_svd
            matrix = cp.asarray([[3., 1.], [0., 2.], [1., -.5]], dtype=cp.float64)
            u, s, vt = exact_svd(matrix)
            cp.cuda.Device().synchronize()
            np.testing.assert_allclose(cp.asnumpy((u * s) @ vt), cp.asnumpy(matrix), atol=1e-12)
            from examples.pzt_era import fit_era, reduced_impulse
            t = np.arange(65) * .01
            h = np.zeros((65, 2)); h[2:] = .98**np.arange(63)[:, None] * [[1., .7]]
            model, _ = fit_era(t, h, 4, backend='cupy')
            rebuilt, _ = reduced_impulse(model, 1, len(t))
            np.testing.assert_allclose(rebuilt, h, atol=1e-11)
            print('PASS: tiny FP64 cuSOLVER SVD and one-state ERA recovery; no full-size MOR computation.')
    print('No wave simulation started.')


if __name__ == '__main__':
    main()
