"""Evaluate saved numerical refinement evidence; no time integration."""
import json
from pathlib import Path
import numpy as np
from traxis.storage import write_json,sha256_file

PROJECT=Path(__file__).resolve().parents[1]


def assess():
    root=PROJECT/'outputs/pzt-development/pzt-readiness'
    selected=json.loads((root/'mesh-32-6-3.json').read_text())
    grid=np.linspace(30000,350000,161)
    def response(d):return np.array(d['attached_response_real'])+1j*np.array(d['attached_response_imag'])
    rows=[]
    for direction,resolution in [('radial',(32,7,3)),('angular',(40,6,3)),('thickness',(32,6,4))]:
        path=root/('mesh-%s-%s-%s.json'%resolution);other=json.loads(path.read_text())
        changes={str(f):float(100*np.linalg.norm((response(selected)-response(other))[abs(grid-f)<=.4*f])/
            np.linalg.norm(response(other)[abs(grid-f)<=.4*f])) for f in (50000,100000,150000,200000,250000)}
        rows.append(dict(direction=direction,resolution=list(resolution),
            free_radial_change_percent=100*abs(selected['free_radial_hz']/other['free_radial_hz']-1),
            attached_band_response_change_percent=changes,raw_sha256=sha256_file(path)))
    passed=all(r['free_radial_change_percent']<1 and max(r['attached_band_response_change_percent'].values())<2 for r in rows)
    answer=dict(passed=passed,selected_resolution=[32,6,3],nodes_per_body=772,
        thresholds=dict(free_radial_change_percent=1,attached_band_response_change_percent=2),
        criteria_note='Engineering readiness targets for an exploratory comparison, not measured physical accuracy or rigorous error bounds.',
        rows=rows,scope='Free disc resonance and electrical driving-point response on a fixed 40x40x1 mm coarse host patch (2x2x1 Q2).',
        frequency_response_note='s=1/350us+i*2pi*f; finite-window exponential weighting, no physical damping added. Uniform samples within +/-40% center frequency.',
        limitations=['Successive refinements do not bound absolute discretization error.',
            'The small host patch is fixed; this does not establish convergence of the full plate or receiver-to-receiver waveform.',
            'Perfect bond, nominal material; not validated training data.'],
        selected_raw_sha256=sha256_file(root/'mesh-32-6-3.json'),
        source_sha256={p:sha256_file(PROJECT/p) for p in ('traxis/pzt_body.py','traxis/piezoelectric.py','examples/pzt_readiness.py',__file__.replace(str(PROJECT)+'\\','').replace('\\','/'))})
    if not passed:raise ArithmeticError('PZT refinement targets failed: '+json.dumps(rows))
    return answer


if __name__=='__main__':
    report=assess();write_json(PROJECT/'docs/evidence/pzt-body-readiness-refinement.json',report)
    print(json.dumps(report,indent=2))
