"""Small matrix parity plus saved six-ring CPU/GPU component voltage parity."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse,json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--local-cuda-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from examples.quadratic_pzt_sparse_check import cuda_paths
    handles=cuda_paths(args.local_cuda_root,args.output)
    import cupy as cp
    from traxis.spectral_piezoelectric import disk_mesh,assemble
    from traxis.pzt_receiver import DiskGeometry
    from traxis.piezoelectric import PiezoMaterial,read_profile
    from examples.spectral_pzt_experiment import run,compare,ROOT
    mat=PiezoMaterial.from_profile(read_profile(ROOT/'configs/pzt/nce51-noliac-ver1404.json'))
    free,total=cp.cuda.runtime.memGetInfo()
    print('GPU:',cp.cuda.runtime.getDeviceProperties(0)['name'], 'free GiB:',free/2**30,flush=True)
    records=[]
    for order in (2,3):
        mesh=disk_mesh(DiskGeometry((.5,.5),.005,.0005,'top',1),side_cells=2,order=order)
        a=assemble(mesh,mat,quadrature_order=6)
        b=assemble(mesh,mat,quadrature_order=6,backend='gpu')
        errors={}
        for key in ('stiffness','mass','coupling','dielectric'):
            x,y=getattr(a,key),getattr(b,key)
            errors[key]=float(np.sqrt((x-y).power(2).sum())/np.sqrt(x.power(2).sum()))
        errors['volume']=abs(a.volume_m3-b.volume_m3)/a.volume_m3
        print('Q'+str(order)+' matrix differences:',errors,flush=True)
        if max(errors.values())>1e-11:raise ArithmeticError('GPU matrix parity failed')
        records.append(dict(order=order,errors=errors))
    result=run(2,16,6,1,args.output,backend='gpu')
    reference=json.loads((ROOT/'outputs/q2-convergence-20261125T183429/rings-6/q2-s16-r6-z1-g6.json').read_text())
    differences=compare(reference,result)
    worst=max(max(v['per_frequency_percent']) for v in differences.values())
    if worst>1e-7:raise ArithmeticError('GPU voltage parity failed')
    receipt=dict(status='passed',device=str(cp.cuda.runtime.getDeviceProperties(0)['name']),
        matrix_checks=records,voltage_comparison=differences,worst_voltage_difference_percent=worst,
        gpu_seconds=result['seconds'],saved_cpu_seconds=reference['seconds'],
        gpu_pool_bytes=cp.get_default_memory_pool().total_bytes(),
        scope='Component frequency response only; FP64, consistent mass, identical quadrature. CPU LU factorization, GPU element integration and triangular solves. Includes first-use CUDA overhead.')
    (args.output/'verification.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
