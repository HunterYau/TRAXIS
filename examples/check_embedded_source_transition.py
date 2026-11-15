"""Compare archived and current CUDA integration on bounded linear systems."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import numpy as np

from examples.embedded_wave_evidence import validate_saved_plan
from examples.tensor_wave import PROJECT, read_json, sealed
from traxis.gpu_enriched_verlet import gpu_explicit_simulate, tensor_patch_probe_set
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json


def verify(reference, output, guard):
    reference,output=Path(reference).resolve(),Path(output).resolve()
    if not output.is_relative_to(PROJECT/'outputs'):
        raise ValueError('source-transition evidence must remain in project outputs')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        scope='Integrator/probe storage equivalence on small SPD systems; no continuum or physical validation.')
    try:
        os.environ['CUPY_CACHE_DIR']=str(output/'kernel-cache')
        import cupy as cp
        plan=read_json(reference/'embedded-wave-plan.json')
        report['source_validation']=validate_saved_plan(plan,reference,guard)
        with zipfile.ZipFile(reference/'source-snapshot.zip') as archive:
            source=archive.read('traxis/gpu_enriched_verlet.py')
        module_name='_traxis_reviewed_archived_verlet'
        module=ModuleType(module_name);sys.modules[module_name]=module
        try:exec(compile(source,str(reference/'source-snapshot.zip')+'/traxis/gpu_enriched_verlet.py','exec'),module.__dict__)
        finally:sys.modules.pop(module_name,None)
        rng=np.random.default_rng(20261115);ns=11;n=3*ns
        matrix=rng.normal(size=(n,n));matrix=matrix.T@matrix+np.eye(n)
        mass=rng.uniform(.8,1.2,n);device_matrix=cp.asarray(matrix)
        class Storage:
            cp=__import__('cupy');lock=nullcontext();device_id=0;applications=0
            def _launch(self,u,out,alpha,beta):
                value=cp.sum(device_matrix*u[None,:],axis=1)
                out[:]=alpha*value+beta*out if beta else alpha*value
                self.applications+=1
        storage=Storage()
        system=SimpleNamespace(mass_diagonal=mass,stable_dt_s=.1,
            stiffness=SimpleNamespace(storage=storage,applications=0),
            screen=dict(device_storage_screen_mib=1.,max_device_mib=256.))
        patches=[SimpleNamespace(indices=np.array([1,4,8]),weights=np.array([.4,-.1,.7]),
            direction=np.array(direction)) for direction in ([0.,0.,1.],[.4,.5,.6],[-1.,.2,.3])]
        dense=np.zeros((3,n))
        for row,patch in zip(dense,patches):row.reshape(3,ns)[:,patch.indices]=patch.direction[:,None]*patch.weights
        sparse=tensor_patch_probe_set(patches,ns);load=rng.normal(size=n);cases=[]
        for initialized in (False,True):
            kwargs=dict(force=lambda t:np.sin(7*t)*np.exp(-t),guard=guard,energy_stride=17)
            if initialized:kwargs.update(initial_displacement=rng.normal(size=n),initial_velocity=rng.normal(size=n))
            old=module.gpu_explicit_simulate(system,512,.01,dense,load,**kwargs)
            new_dense=gpu_explicit_simulate(system,512,.01,dense,load,**kwargs)
            new_sparse=gpu_explicit_simulate(system,512,.01,sparse,load,**kwargs)
            errors={}
            for name in ('time_s','displacement_m','energy_time_s','physical_energy_j','modified_energy_j',
                         'final_displacement_m','final_velocity_m_s'):
                baseline=getattr(old,name)
                for label,result in [('dense',new_dense),('sparse',new_sparse)]:
                    errors[label+'_'+name]=float(np.max(abs(getattr(result,name)-baseline)))
            passed=max(errors.values())<1e-12
            cases.append(dict(nonzero_initial_state=initialized,maximum_absolute_errors=errors,passed=passed))
        guard.check(force=True)
        report.update(completed=True,passed=all(c['passed'] for c in cases),cases=cases,
            reference_plan_fingerprint=plan['fingerprint'],archived_integrator_sha256=plan['source_sha256']['gpu_enriched_verlet.py'],
            current_integrator_sha256=sha256_file(PROJECT/'traxis/gpu_enriched_verlet.py'),checker_sha256=sha256_file(__file__))
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['resources']=guard.report();write_json(output/'source-transition-check.json',sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    guard=ResourceGuard(ResourceLimits(max_seconds=90,max_rss_mib=1024,max_system_memory_percent=89.9,min_available_mib=768))
    report=verify(args.reference_run,args.output,guard)
    print(json.dumps({k:v for k,v in report.items() if k!='resources'}));return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
