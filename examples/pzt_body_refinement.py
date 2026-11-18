"""Free-disc refinement diagnostic; not a plate/adhesive validation."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from traxis.piezoelectric import PiezoMaterial,read_profile,assemble_piezoelectric,electrode_port
from traxis.pzt_body import disk_mesh
from traxis.pzt_receiver import DiskGeometry
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import write_json,sha256_file


def check():
    project=Path(__file__).resolve().parents[1]
    guard=ResourceGuard(ResourceLimits(max_seconds=240,max_rss_mib=2048))
    material=PiezoMaterial.from_profile(read_profile(project/'configs/pzt/nce51-noliac-ver1404.json'))
    geometry=DiskGeometry((0.,0.),.005,.0005,'top',1)
    rows=[];frequencies=np.arange(50000,300001,50000)
    for sectors,rings,layers in ((16,1,2),(24,2,2),(32,3,2),(48,3,2)):
        guard.check(force=True)
        mesh,nface=disk_mesh(geometry,sectors=sectors,rings=rings,layers=layers)
        port=electrode_port(assemble_piezoelectric(mesh,material,guard=guard),
            np.arange(nface),np.arange(nface)+layers*nface)
        eigenvalues,modes=eigh(port.stiffness_sc,port.mass)
        participation=(port.coupling_n_v@modes)**2
        hz=np.sqrt(np.maximum(eigenvalues,0))/(2*np.pi)
        selected=np.flatnonzero((hz>20000)&(hz<500000))
        dominant=selected[np.argmax(participation[selected]/eigenvalues[selected])]
        dynamic=[float(port.capacitance_f+np.sum(participation/(eigenvalues-(2*np.pi*f)**2))) for f in frequencies]
        row=dict(sectors=sectors,rings=rings,layers=layers,nodes=len(mesh.points_m),
            dominant_electrically_excited_resonance_hz=float(hz[dominant]),
            clamped_capacitance_f=port.capacitance_f,frequencies_hz=frequencies.tolist(),
            lossless_dynamic_capacitance_f=dynamic)
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(rows=rows,scope='Free nominal NCE51 disc only; no plate, bond or instrument; lossless response',
        final_resonance_change_percent=100*abs(rows[-1]['dominant_electrically_excited_resonance_hz']/rows[-2]['dominant_electrically_excited_resonance_hz']-1),
        analysis_sha256=sha256_file(Path(__file__)),ready_for_training_data=False)
    write_json(project/'docs/evidence/pzt-body-free-disc-refinement-20261117.json',report)
    return report


if __name__=='__main__':check()
