"""Select tensor Lamb spectral support from analytic displacement profiles."""
import argparse
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import sys
import zipfile
import numpy as np
from scipy import sparse
from scipy.integrate import quad
from scipy.linalg import eigh
from examples.tensor_reference import parity_projection
from traxis.config import Material,Plate
from traxis.lamb_profiles import fundamental_profile,spectral_support
from traxis.tensor_hex import make_grid,assemble_tensor
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.highorder_verification import distribution_identity
from traxis.storage import canonical_hash,sha256_file,write_json
from traxis.resources import ResourceGuard,ResourceLimits


def burst_transform(frequency_hz,carrier_hz=100000.,cycles=2):
    """Fourier transform of sin(carrier*t) times the finite Hann window."""
    f = np.asarray(frequency_hz)
    if np.iscomplexobj(f) or not np.isfinite(f).all() or not math.isfinite(carrier_hz) or carrier_hz<=0 or cycles!=2:
        raise ValueError('bounded burst transform needs real frequencies and two positive-carrier cycles')
    duration = cycles/carrier_hz
    def integral(offset):
        return duration*np.exp(-1j*np.pi*offset*duration)*np.sinc(offset*duration)
    def hann(offset):
        return .5*integral(offset)-.25*(integral(offset-1/duration)+integral(offset+1/duration))
    return (hann(f-carrier_hz)-hann(f+carrier_hz))/(2j)


def burst_band(burst):
    carrier,duration = burst['carrier_hz'],burst['cycles']/burst['carrier_hz']
    lower,upper = burst['band_hz']
    time_energy = quad(lambda t:(np.sin(2*np.pi*carrier*t)*.5*(1-np.cos(2*np.pi*t/duration)))**2,
                       0,duration,epsabs=1e-16,epsrel=1e-12)[0]
    analytic_time_energy = 3*duration/16
    if abs(time_energy/analytic_time_energy-1)>1e-12:
        raise ArithmeticError('time-domain burst energy differs from the independent analytic integral')
    retained,error = quad(lambda f:float(abs(burst_transform(f,carrier,2))**2),lower,upper,epsabs=1e-16,epsrel=1e-11)
    fraction = retained/(time_energy/2)
    return dict(time_energy_n2_s=time_energy,positive_frequency_energy_n2_s=time_energy/2,
        retained_force_energy_fraction=fraction,quadrature_absolute_error=error,
        passed=bool(fraction>=burst['minimum_force_spectral_energy_fraction']),
        scope='Nominal mechanical force spectrum only; receiver transfer functions and PZT electrical drive are not included.')


def identities():
    names=('tensor_profile_study.py','tensor_reference.py')
    return dict(source_sha256=reference_source_hashes(),driver_sha256={n:sha256_file(Path(__file__).parent/n) for n in names})


def checked_configuration(config):
    expected=dict(kind='tensor_lamb_profile_design',orders=[2,3,4],
        lateral_spacing_m=[.005,.005/1.5,.0025],thickness_m=[.003,.001],thickness_cells=[1,2,4],
        frequencies_hz=[5000,10000,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000],
        modes=['A0','S0'],stiffness_rules=['gll','exact'],profile_mass_coverage=.99,
        maximum_support_relative_frequency_error=.0005,maximum_profile_projection_error=1e-10,
        burst=dict(carrier_hz=100000,cycles=2,band_hz=[5000,250000],minimum_force_spectral_energy_fraction=.999))
    if config!=expected:
        raise ValueError('profile study requires its fixed nonempty design and unchanged criteria')
    return config


def study(config,output,guard):
    checked_configuration(config)
    material = Material(); rows=[]; profiles={}; spectral=burst_band(config['burst'])
    for thickness in config['thickness_m']:
        for frequency in config['frequencies_hz']:
            for mode in config['modes']:
                profile=fundamental_profile(frequency,thickness,mode,material)
                points=np.linspace(-thickness/2,thickness/2,65)
                u=profile.fields(points)[0]
                traction=profile.traction(np.array([-thickness/2,thickness/2]))
                relative=float(np.linalg.norm(traction)/(material.lame[1]*profile.wavenumber_rad_m*np.linalg.norm(u)))
                residual=profile.navier_relative_residual(points)
                if max(relative,residual)>1e-8:
                    raise ArithmeticError('analytic comparison profile fails free tractions or the elastic PDE')
                profiles[thickness,frequency,mode]=(profile,relative,residual)
    proof_path=output/'profile-proofs.npz'
    with zipfile.ZipFile(proof_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        def save(key,value):
            stream=io.BytesIO();np.save(stream,np.asarray(value),allow_pickle=False);archive.writestr(key+'.npy',stream.getvalue())
        for thickness in config['thickness_m']:
            for width in config['lateral_spacing_m']:
                for order in config['orders']:
                    for depth in config['thickness_cells']:
                        guard.check(force=True)
                        grid=make_grid(Plate(width,width,thickness),(1,1,depth),order)
                        for rule in config['stiffness_rules']:
                            system=assemble_tensor(grid,material,rule,guard)
                            diagonal=sparse.diags(system.mass_kg)
                            for frequency in config['frequencies_hz']:
                                for mode in config['modes']:
                                    guard.check()
                                    profile,traction_residual,pde_residual=profiles[thickness,frequency,mode]
                                    wave=profile.wavenumber_rad_m
                                    projection=parity_projection(grid,wave,mode)
                                    k=(projection.conj().T@system.stiffness@projection).toarray()
                                    m=(projection.conj().T@diagonal@projection).toarray()
                                    if max(np.linalg.norm(a-a.conj().T)/np.linalg.norm(a) for a in (k,m))>1e-10:
                                        raise ArithmeticError('profile comparison lost Hermitian structure')
                                    if np.linalg.norm(m-np.diag(m.diagonal()))>1e-12*np.linalg.norm(m):
                                        raise ArithmeticError('profile reduction no longer has diagonal nodal mass')
                                    u=profile.fields(grid.points_m[:,2]-thickness/2)[0]
                                    target=(u*np.exp(1j*wave*grid.points_m[:,0])[None,:]).ravel()
                                    reference=(projection.conj().T@(system.mass_kg*target))/m.diagonal()
                                    projected=projection@reference
                                    projection_error=float(np.sqrt(np.vdot(projected-target,system.mass_kg*(projected-target)).real/
                                        np.vdot(target,system.mass_kg*target).real))
                                    if projection_error>config['maximum_profile_projection_error']:
                                        raise ArithmeticError('analytic nodal profile is absent from the periodic/parity space')
                                    eigenvalues,eigenvectors=eigh(k,m)
                                    if np.any(eigenvalues<=0):
                                        raise ArithmeticError('profile comparison has a nonpositive eigenvalue')
                                    selected,overlaps=spectral_support(eigenvectors,m,reference,config['profile_mass_coverage'])
                                    frequencies=np.sqrt(eigenvalues)/(2*np.pi)
                                    errors=frequencies/frequency-1
                                    dominant=int(np.argmax(overlaps))
                                    worst=float(np.max(abs(errors[selected])))
                                    key=f'case{len(rows)}'
                                    row=dict(thickness_m=thickness,lateral_spacing_m=width,order=order,thickness_cells=depth,
                                        stiffness_rule=rule,target_frequency_hz=frequency,mode=mode,wavenumber_rad_m=wave,
                                        phase_rad=wave*width,outside_first_zone=bool(wave*width>=np.pi),
                                        profile_projection_relative_error=projection_error,analytic_traction_relative_residual=traction_residual,
                                        analytic_navier_relative_residual=pde_residual,dominant_index=dominant,
                                        dominant_profile_mass_fraction=float(overlaps[dominant]),dominant_signed_relative_frequency_error=float(errors[dominant]),
                                        selected_modes=selected.tolist(),selected_profile_mass_fraction=float(overlaps[selected].sum()),
                                        maximum_support_relative_frequency_error=worst,
                                        passed=bool(worst<config['maximum_support_relative_frequency_error']),proof_key=key)
                                    rows.append(row)
                                    for label,value in dict(eigenvalues=eigenvalues,profile_overlaps=overlaps,selected_modes=selected,
                                        selected_eigenvectors=eigenvectors[:,selected],reference_coefficients=reference,
                                        target_displacement=target,projected_mass_diagonal=m.diagonal(),
                                        profile_z_m=grid.points_m[:,2]-thickness/2).items():
                                        save(key+'_'+label,value)
    finest=[c for c in rows if c['order']==4 and c['thickness_cells']==4 and c['lateral_spacing_m']==.0025]
    if len(rows)!=2592 or len(finest)!=96:
        raise ArithmeticError('profile study did not produce every declared case')
    return dict(cases=rows,burst_spectrum=spectral,archive_sha256=sha256_file(proof_path),
        passed=bool(spectral['passed'] and all(c['passed'] for c in finest)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--configuration',type=Path,default=Path('configs/tensor-profile-study.json'))
    parser.add_argument('--worker',action='store_true');parser.add_argument('--plan',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    guard=ResourceGuard(ResourceLimits(max_seconds=180,max_rss_mib=512,max_system_memory_percent=85,min_available_mib=2048))
    if args.worker:
        plan=json.loads(args.plan.read_text())
        if (canonical_hash({k:v for k,v in plan.items() if k!='fingerprint'})!=plan['fingerprint']
                or any(plan[k]!=v for k,v in identities().items()) or distribution_identity(guard)!=plan['runtime']):
            raise ValueError('profile study source/runtime/plan changed')
        checked_configuration(plan['configuration'])
        output.mkdir()
        report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,plan_fingerprint=plan['fingerprint'])
        try:
            report.update(study(plan['configuration'],output,guard))
            if any(plan[k]!=v for k,v in identities().items()) or distribution_identity(guard)!=plan['runtime']:
                raise ValueError('profile study source changed during verification')
            report['completed']=True
        except Exception as error:
            report['error']=f'{type(error).__name__}: {error}';raise
        finally:
            report['resources']=guard.report();report['fingerprint']=canonical_hash(report)
            write_json(output/'profile-report.json',report)
        return 0 if report['passed'] else 2
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('profile study needs a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    config=checked_configuration(json.loads(args.configuration.read_text()))
    plan={**identities(),'configuration':config,'configuration_sha256':sha256_file(args.configuration),
        'material':asdict(Material()),'runtime':distribution_identity(guard),'limits':guard.report()['limits']}
    with zipfile.ZipFile(output/'source-snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in plan['source_sha256']:
            archive.write(Path(__file__).resolve().parents[1]/'traxis'/name,'traxis/'+name)
        for name in plan['driver_sha256']:
            archive.write(Path(__file__).parent/name,'examples/'+name)
    plan['source_snapshot_sha256']=sha256_file(output/'source-snapshot.zip');plan['fingerprint']=canonical_hash(plan)
    path=output/'profile-plan.json';write_json(path,plan)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    temporary=output/'temporary-files';temporary.mkdir();env.update({name:str(temporary) for name in ('TEMP','TMP','TMPDIR')})
    command=[sys.executable,'-m','examples.tensor_profile_study','--worker','--plan',str(path),'--output',str(output/'native')]
    report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,plan_fingerprint=plan['fingerprint'])
    try:
        worker=run_monitored(command,Path(__file__).resolve().parents[1],env,guard,output/'worker.log');report['worker_run']=worker
        if worker['exit_code'] not in (0,2) or not (output/'native/profile-report.json').is_file():
            raise RuntimeError(f"profile worker exited {worker['exit_code']}; see worker.log")
        native=json.loads((output/'native/profile-report.json').read_text())
        if (native['plan_fingerprint']!=plan['fingerprint'] or canonical_hash({k:v for k,v in native.items() if k!='fingerprint'})!=native['fingerprint']
                or native['archive_sha256']!=sha256_file(output/'native/profile-proofs.npz')
                or any(plan[k]!=v for k,v in identities().items())):
            raise ValueError('profile study receipt or source changed')
        report.update(completed=True,passed=worker['exit_code']==0 and native['passed'],native_report_sha256=sha256_file(output/'native/profile-report.json'))
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
    finally:
        report['resources']=guard.report();report['fingerprint']=canonical_hash(report);write_json(output/'monitor-result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='resources'}))
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
