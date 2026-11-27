"""Separate matrix-free native-SpSM adapter for the unchanged graph executor.

Creates isolated function globals to substitute ONLY the operator factory.
The existing validation and recurrence execute their original code objects;
no monkeypatching, solver source edits or alternate integration equations.
"""
from types import FunctionType
import hashlib
import numpy as np
from traxis import gpu_lts_optimized as executor
from traxis.gpu_native_cusparse import NativeLibrary,NativeCachedLU
from traxis.gpu_spectral_matrixfree import couple_matrixfree


def couple_native(base,attachments,masses,*,guard=None,allow_fallback=False):
    system=couple_matrixfree(base,attachments,masses,guard=guard)
    patches=system.stiffness.storage.device_patches
    components={id(p.component):p.component for p in patches}.values()
    owners=[(c,'electrical') for c in components]+[(p,'factor') for p in patches]
    replacements=[]
    try:
        library=NativeLibrary()
        for owner,name in owners:
            old=getattr(owner,name);new=NativeCachedLU(old,library)
            replacements.append((owner,name,new))
        extra=sum(n.bytes()-n.fallback.bytes() for _,_,n in replacements)
        cp=system.stiffness.storage.cp
        if extra+system.screen['device_storage_screen_mib']*2**20>system.screen['max_device_mib']*2**20:
            raise MemoryError('Native analysis workspace exceeds device budget')
        for owner,name,new in replacements:setattr(owner,name,new)
    except BaseException as error:
        for _,_,new in replacements:new.close(close_fallback=False)
        if not allow_fallback or not isinstance(error,Exception):
            system.close();raise
        system.native_factors=[]
        system.screen.update(native_bridge=False,cuda_graph_capture_supported=False,native_unavailable_reason=str(error))
        return system
    system.native_factors=[n for _,_,n in replacements]
    system.screen.update(native_bridge=True,cuda_graph_capture_supported=True,native_library=library.identity,
        pzt_storage='matrixfree-tensor-native-cached-lu',
        native_analysis_calls=sum(n.analysis_calls for n in system.native_factors),
        native_extra_workspace_bytes=extra,
        device_storage_screen_mib=system.screen['device_storage_screen_mib']+extra/2**20)
    def set_native_enabled(enabled):
        if type(enabled) is not bool:raise ValueError('Boolean native selection required')
        cp=system.stiffness.storage.cp
        if cp.cuda.runtime.streamIsCapturing(cp.cuda.get_current_stream().ptr):
            raise RuntimeError('Select fallback before capture/execution')
        for plan in system.native_factors:
            plan.stream.synchronize();plan.enabled=enabled
        system.screen['cuda_graph_capture_supported']=enabled
    system.set_native_enabled=set_native_enabled
    return system


class MatrixFreeReusableActions(executor.ReusableActions):
    def __init__(self,system,fine):
        factors=getattr(system,'native_factors',())
        if not factors or not all(f.cuda_graph_capture_supported for f in factors):
            raise ValueError('Native factors required; use original LocalActions for cached CuPy fallback')
        # Executor creates its nonblocking stream before constructing actions.
        for factor in factors:factor.bind_stream()
        super().__init__(system,fine)

    def dense(self,action,x,out):
        """Existing executor hook; action is implicit, never a dense matrix."""
        if not callable(getattr(action,'action',None)):
            raise TypeError('Expected an implicit PZT action')
        self.storage.cp.copyto(out,self.storage.matvec(action,x))


def _clone(function,**replacements):
    namespace=dict(function.__globals__);namespace.update(replacements)
    result=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=dict(function.__kwdefaults__ or {})
    return result


_execute=_clone(executor._execute,ReusableActions=MatrixFreeReusableActions)
_simulate=_clone(executor.gpu_lts_simulate,_execute=_execute)


def gpu_lts_simulate(system,*args,**kwargs):
    """Original p=2 executor with native matrix-free actions, graphs optional.

Uses the existing explicitly unchecked study policy. A compatible matrix-free
stability screen is separate work; never silently disables a requested screen.
"""
    if kwargs.get('substeps',2)!=2:raise ValueError('Native graph adapter supports existing p=2 only')
    if kwargs.get('screen_stability',True):
        raise NotImplementedError('Native implicit adapter requires an explicitly selected compatible stability policy')
    factors=getattr(system,'native_factors',())
    if not factors or not all(f.cuda_graph_capture_supported for f in factors):
        raise ValueError('Native bridge unavailable/disabled; use the preserved original executor with cached fallback')
    with system.stiffness.storage.lock:
        try:
            t,y,stats=_simulate(system,*args,**kwargs)
            stats.update(execution_backend='spectral-matrixfree-native',native_library=system.screen['native_library'],
                native_analysis_calls=sum(f.analysis_calls for f in factors),
                original_executor_bytecode_sha256=hashlib.sha256(executor._execute.__code__.co_code).hexdigest())
            return t,y,stats
        finally:
            # Restore binding for ordinary subsequent calls in the caller stream.
            for factor in factors:factor.bind_stream()
