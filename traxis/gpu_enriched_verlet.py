"""Device-resident explicit Verlet for the checked enriched elastic operator."""

from dataclasses import dataclass
import math

import numpy as np

from traxis.config import positive


PROBE_KERNEL = r'''
extern "C" __global__ void sample_probes(
    const int* starts, const int* indices, const double* weights, const double* u,
    double* output, int probes, int sample) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if (p>=probes) return;
    double value=0.;
    for (int j=starts[p];j<starts[p+1];++j) value+=weights[j]*u[indices[j]];
    output[(long long)sample*probes+p]=value;
}
'''


@dataclass
class GPUExplicitResult:
    time_s: np.ndarray
    displacement_m: np.ndarray
    energy_time_s: np.ndarray
    physical_energy_j: np.ndarray
    modified_energy_j: np.ndarray
    final_displacement_m: np.ndarray
    final_velocity_m_s: np.ndarray
    statistics: dict


def gpu_explicit_simulate(system,steps,dt_s,probes,load,*,force=None,initial_displacement=None,
                          initial_velocity=None,output_stride=1,energy_stride=16,guard=None,
                          progress=None,max_workspace_mib=512):
    """Run the requested interval in float64; never shorten it to fit a guard.

    Probes are compressed without dropping nonzero coefficients. Energy is
    sampled independently of traces. Modified energy is an unforced invariant;
    forced-interval energy is retained without asserting an exact work identity.
    """
    positive('explicit time step',dt_s)
    if (type(steps) is not int or not 1<=steps<=500000 or type(output_stride) is not int
            or not 1<=output_stride<=64 or steps%output_stride or type(energy_stride) is not int
            or not 1<=energy_stride<=1024 or (progress is not None and not callable(progress))):
        raise ValueError('invalid explicit step count, strides or progress callback')
    if dt_s>.95*system.stable_dt_s:
        raise ValueError('time step exceeds the conservative enriched explicit stability limit')
    mass,probes,load = (np.asarray(v) for v in (system.mass_diagonal,probes,load))
    n = len(mass)
    if (mass.shape!=(n,) or np.iscomplexobj(mass) or not np.isfinite(mass).all() or np.any(mass<=0)
            or np.iscomplexobj(probes) or probes.ndim!=2 or probes.shape[1]!=n or not 1<=len(probes)<=32
            or not np.isfinite(probes).all() or np.iscomplexobj(load) or load.shape!=(n,) or not np.isfinite(load).all()
            or not math.isfinite(max_workspace_mib) or not 0<max_workspace_mib<=2048):
        raise ValueError('explicit states need matching finite real mass/load/probe data')
    samples = steps//output_stride+1
    energy_steps = np.r_[np.arange(0,steps+1,energy_stride),steps] if steps%energy_stride else np.arange(0,steps+1,energy_stride)
    indices = [np.flatnonzero(row).astype(np.int32) for row in probes]
    starts = np.r_[0,np.cumsum([len(i) for i in indices])].astype(np.int32)
    if starts[-1]>100000:
        raise MemoryError('explicit probe support exceeds 100000 coefficients')
    sample_bytes = samples*len(probes)*8+len(energy_steps)*24
    workspace_mib = (16*n*8+sample_bytes+starts[-1]*16)/1024**2
    if sample_bytes>64*1024**2 or workspace_mib>max_workspace_mib:
        raise MemoryError('explicit vector/sample storage exceeds its declared screen')
    def vector(value):
        if value is None:
            return np.zeros(n)
        value = np.asarray(value)
        if value.shape!=(n,) or np.iscomplexobj(value) or not np.isfinite(value).all():
            raise ValueError('explicit initial states must be finite real coefficient vectors')
        return value.astype(float,copy=True)
    initial_u,initial_v = vector(initial_displacement),vector(initial_velocity)
    def amplitude(time_s):
        value = float(force(time_s)) if force else 0.
        if not math.isfinite(value):
            raise ValueError('explicit force amplitude is nonfinite')
        return value
    storage = system.stiffness.storage
    if (system.screen['device_storage_screen_mib']+(sample_bytes+starts[-1]*16)/1024**2
            >system.screen['max_device_mib']):
        raise MemoryError('explicit samples/probes exceed the declared total device budget')
    cp = storage.cp
    physical,modified = np.empty(len(energy_steps)),np.empty(len(energy_steps))
    with storage.lock,cp.cuda.Device(storage.device_id):
        if guard:
            guard.check(force=True)
        free,_ = cp.cuda.runtime.memGetInfo()
        if workspace_mib*1024**2>.85*free:
            raise MemoryError('explicit workspace exceeds 85% of currently free VRAM')
        u,v,device_mass,device_load = (cp.asarray(a,dtype=cp.float64) for a in (initial_u,initial_v,mass,load))
        ku,next_a = cp.empty(n,dtype=cp.float64),cp.empty(n,dtype=cp.float64)
        d_starts = cp.asarray(starts)
        d_indices = cp.asarray(np.concatenate(indices) if starts[-1] else np.empty(0,dtype=np.int32))
        d_weights = cp.asarray(np.concatenate([row[i] for row,i in zip(probes,indices)]) if starts[-1] else np.empty(0),dtype=cp.float64)
        traces = cp.empty((samples,len(probes)),dtype=cp.float64)
        sample_kernel = cp.RawKernel(PROBE_KERNEL,'sample_probes',options=('--std=c++11',))
        applications_before = storage.applications
        def action():
            storage._launch(u,ku,1.,0.)
            system.stiffness.applications += 1
        def sample(index):
            sample_kernel((1,),(64,),(d_starts,d_indices,d_weights,u,traces,np.int32(len(probes)),np.int32(index)))
        def energy(index):
            physical[index] = float((cp.sum(device_mass*v*v+u*ku,dtype=cp.float64)*.5).item())
            modified[index] = physical[index]-dt_s**2/8*float(cp.sum(ku*ku/device_mass,dtype=cp.float64).item())
            if not math.isfinite(physical[index]) or not math.isfinite(modified[index]):
                raise ArithmeticError('explicit GPU energy is nonfinite')
        action()
        acceleration = (amplitude(0.)*device_load-ku)/device_mass
        sample(0)
        energy(0)
        energy_index = 1
        for step in range(1,steps+1):
            u += dt_s*v+.5*dt_s*dt_s*acceleration
            action()
            cp.subtract(amplitude(step*dt_s)*device_load,ku,out=next_a)
            next_a /= device_mass
            v += .5*dt_s*(acceleration+next_a)
            acceleration,next_a = next_a,acceleration
            if step%output_stride==0:
                sample(step//output_stride)
            if energy_index<len(energy_steps) and step==energy_steps[energy_index]:
                energy(energy_index)
                energy_index += 1
            if step%16==0 or step==steps:
                cp.cuda.get_current_stream().synchronize()
                if guard:
                    guard.check()
                if progress:
                    progress(dict(steps_completed=step,requested_steps=steps,time_s=step*dt_s,
                        requested_duration_s=steps*dt_s,stiffness_applications=storage.applications-applications_before))
        answer,final_u,final_v = cp.asnumpy(traces),cp.asnumpy(u),cp.asnumpy(v)
        if not all(np.isfinite(a).all() for a in (answer,final_u,final_v)):
            raise ArithmeticError('explicit GPU trace or final state is nonfinite')
        if guard:
            guard.check(force=True)
        stats = dict(method='gpu_velocity_verlet',precision='float64',steps_completed=steps,
            requested_steps=steps,duration_s=steps*dt_s,workspace_screen_mib=workspace_mib,
            probe_nonzero_coefficients=int(starts[-1]),stiffness_applications=storage.applications-applications_before,
            per_step_host_state_transfers=0,final_host_state_downloads=2)
    return GPUExplicitResult(np.arange(samples)*dt_s*output_stride,answer,energy_steps*dt_s,
        physical,modified,final_u,final_v,stats)
