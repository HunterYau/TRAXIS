"""Surface quadrature check between PZT tie nodes; no system solve."""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from traxis.spectral_piezoelectric import disk_mesh
from traxis.pzt_receiver import DiskGeometry
from traxis.tensor_hex import line_rule
from traxis.tensor_patches import _basis
from traxis.lamb_profiles import fundamental_profile
from traxis.config import Material


def main():
    root=Path(__file__).resolve().parents[1];out=root/'outputs/q2-interface-audit-002'
    mesh=disk_mesh(DiskGeometry((.5,.5),.005,.0005,'top',1),radial_cells=8)
    face=mesh.points_m[mesh.hexahedra[:,::3],:2]
    x,w=leggauss(4);uv=np.array(np.meshgrid(x,x,indexing='ij')).reshape(2,-1).T
    basis=_basis(line_rule(2)[0]);N=np.array([np.outer([p(a) for p in basis],[p(b) for p in basis]).ravel() for a,b in uv])
    gradients=np.array([[np.outer([p.deriv()(a) for p in basis],[p(b) for p in basis]).ravel(),np.outer([p(a) for p in basis],[p.deriv()(b) for p in basis]).ravel()] for a,b in uv])
    xy=N@face;J=gradients[None]@face[:,None];weight=np.linalg.det(J)*np.outer(w,w).ravel()
    bases=_basis(line_rule(5)[0]);rows=[]
    def host_trace(x,k,cells):
        cell=np.minimum(np.floor(x*cells).astype(int),cells-1);local=2*(x*cells-cell)-1
        nodes=(cell[...,None]+(line_rule(5)[0]+1)/2)/cells
        return np.sum(np.array([p(local) for p in bases]).transpose(*range(1,local.ndim+1),0)*np.exp(1j*k*nodes),axis=-1)
    for cells in (250,500,1000):
        for f in (50000,100000,150000,200000,250000,300000):
            for mode in ('S0','A0'):
                k=fundamental_profile(f,.001,mode,Material(young_pa=71e9,poisson=.33,density_kg_m3=2700)).wavenumber_rad_m
                nodal=host_trace(face[:,:,0]+.5,k,cells)
                slave=nodal@N.T;master=host_trace(xy[:,:,0]+.5,k,cells)
                exact=np.exp(1j*k*(xy[:,:,0]+.5))
                norm=np.sqrt(np.sum(weight*abs(master)**2))
                rows.append(dict(cells=cells,frequency_hz=f,mode=mode,
                    integrated_tie_trace_gap_percent=float(100*np.sqrt(np.sum(weight*abs(slave-master)**2))/norm),
                    integrated_master_interpolation_error_percent=float(100*np.sqrt(np.sum(weight*abs(master-exact)**2))/norm)))
    (out/'surface-quadrature.json').write_text(json.dumps(rows,indent=2));print('Surface quadrature trace audit saved')


if __name__=='__main__':main()
