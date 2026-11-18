"""Bounded mesh studies for a nominal disc and a disc attached to a small plate.

Never runs the five full-plate experiments. Frequency responses are evaluated
in the right Laplace half-plane, alpha=1/350us, to avoid undamped poles. This
is finite-window exponential weighting, not added physical damping.
"""
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh

from traxis.config import Material,Plate
from traxis.gpu_tensor import make_layout,mass_diagonal,streamed_action
from traxis.piezoelectric import PiezoMaterial,read_profile,assemble_piezoelectric,electrode_port
from traxis.pzt_body import attach_disk,disk_mesh
from traxis.pzt_receiver import DiskGeometry
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import write_json,sha256_file

PROJECT=Path(__file__).resolve().parents[1]


def modes_response(k,m,g,cap,frequencies):
    values,vectors=eigh((k+k.T)*.5,m,check_finite=True)
    participation=(g@vectors)**2
    s=1/.00035+2j*np.pi*frequencies
    response=cap+np.sum(participation[:,None]/(values[:,None]+s[None,:]**2),axis=0)
    hz=np.sqrt(np.maximum(values,0))/(2*np.pi)
    selected=np.flatnonzero((hz>20000)&(hz<500000))
    dominant=selected[np.argmax(participation[selected]/values[selected])]
    return float(hz[dominant]),response


def main():
    guard=ResourceGuard(ResourceLimits(max_seconds=1200,max_rss_mib=3072))
    material=PiezoMaterial.from_profile(read_profile(PROJECT/'configs/pzt/nce51-noliac-ver1404.json'))
    geometry=DiskGeometry((.02,.02),.005,.0005,'top',1)
    layout=make_layout(Plate(.04,.04,.001),(2,2,1),2)
    nh=layout.screen['vector_dofs'];host=np.zeros((nh,nh))
    for j in range(nh):
        u=np.zeros(nh);u[j]=1.;host[:,j]=streamed_action(layout,Material(71e9,.33,2700),u)
    hm=mass_diagonal(layout,Material(71e9,.33,2700))
    frequencies=np.linspace(30000,350000,161)
    rows=[];started=time.perf_counter()
    for resolution in ((32,6,3),(32,7,3),(40,6,3),(32,6,4)):
        guard.check(force=True);sectors,rings,layers=resolution
        path=PROJECT/f'outputs/pzt-readiness/mesh-{sectors}-{rings}-{layers}.json'
        if path.exists():
            rows.append(json.loads(path.read_text()));continue
        mesh,nface=disk_mesh(geometry,sectors=sectors,rings=rings,layers=layers)
        port=electrode_port(assemble_piezoelectric(mesh,material,guard=guard),np.arange(nface),np.arange(nface)+layers*nface)
        radial,free_response=modes_response(port.stiffness_sc,port.mass,port.coupling_n_v,port.capacitance_f,frequencies)
        del port
        body=attach_disk(layout,material,geometry,electrical='voltage',sectors=sectors,rings=rings,layers=layers,guard=guard,max_workspace_mib=2048)
        n=body.next_dof;k=np.zeros((n,n));k[:nh,:nh]=host
        m=np.diag(np.r_[hm,np.zeros(n-nh)]);ix=np.ix_(body.indices,body.indices)
        k[ix]+=body.stiffness;m[ix]+=body.mass;g=np.zeros(n);g[body.indices]=body.coupling
        _,attached_response=modes_response(k,m,g,body.capacitance_f,frequencies)
        row=dict(resolution=list(resolution),nodes=len(mesh.points_m),free_radial_hz=radial,
            capacitance_f=body.capacitance_f,free_response_real=free_response.real.tolist(),
            free_response_imag=free_response.imag.tolist(),attached_response_real=attached_response.real.tolist(),
            attached_response_imag=attached_response.imag.tolist(),elapsed_seconds=time.perf_counter()-started)
        rows.append(row)
        write_json(PROJECT/f'outputs/pzt-readiness/mesh-{sectors}-{rings}-{layers}.json',row)
        print(json.dumps({key:row[key] for key in ('resolution','nodes','free_radial_hz','capacitance_f','elapsed_seconds')}),flush=True)
        del k,m,body
    write_json(PROJECT/'outputs/pzt-readiness/fine-study.json',dict(rows=rows,frequencies_hz=frequencies.tolist()))


if __name__=='__main__':main()
