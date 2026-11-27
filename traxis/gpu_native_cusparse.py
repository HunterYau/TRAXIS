"""Narrow CUDA-12 cuSPARSE ctypes bridge with cached CuPy LU fallback.

Only attaches to an already loaded cuSPARSE library. Never guesses a toolkit
path or loads a second CUDA major. Own handles are bound outside capture.
Native NVIDIA SpSM_solve has TEN arguments: no externalBuffer parameter.
"""
import ctypes as ct
import os
from pathlib import Path
import sys
import numpy as np

P=ct.c_void_p
I=ct.c_int
SP_SM_ARGS=(P,I,I,P,P,P,P,I,I,P)


def loaded_library_path():
    if sys.platform=='win32':
        kernel=ct.WinDLL('kernel32',use_last_error=True)
        kernel.GetModuleHandleW.argtypes=[ct.c_wchar_p];kernel.GetModuleHandleW.restype=P
        handle=kernel.GetModuleHandleW('cusparse64_12.dll')
        if not handle:raise RuntimeError('CuPy has not loaded CUDA-12 cuSPARSE')
        kernel.GetModuleFileNameW.argtypes=[P,ct.c_wchar_p,ct.c_uint]
        kernel.GetModuleFileNameW.restype=ct.c_uint
        name=ct.create_unicode_buffer(32768)
        if not kernel.GetModuleFileNameW(handle,name,len(name)):
            raise OSError(ct.get_last_error(),'Cannot resolve loaded cuSPARSE path')
        return str(Path(name.value).resolve()),int(handle)
    if sys.platform.startswith('linux'):
        paths=set()
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields=line.split(maxsplit=5)
            if len(fields)==6 and 'libcusparse.so.12' in fields[5]:
                paths.add(str(Path(fields[5]).resolve()))
        if len(paths)!=1:raise RuntimeError('Expected exactly one loaded CUDA-12 cuSPARSE library')
        return paths.pop(),None
    raise RuntimeError('Native bridge currently supports Windows/Linux x86_64 only')


class NativeLibrary:
    def __init__(self):
        import cupy as cp
        from cupyx import cusparse as cs
        if ct.sizeof(P)!=8:raise RuntimeError('64-bit native CUDA ABI required')
        self.cp=cp;self.runtime=int(cp.cuda.runtime.runtimeGetVersion())
        if self.runtime//1000!=12:raise RuntimeError('This verified bridge targets CUDA runtime 12 only')
        cupy_handle=cs._device.get_cusparse_handle()  # Loads the matching library first.
        expected=int(cs._cusparse.getVersion(cupy_handle))
        self.path,loaded_handle=loaded_library_path()
        if sys.platform=='win32':
            self.dll=ct.WinDLL(self.path)
            if self.dll._handle!=loaded_handle:raise RuntimeError('cuSPARSE library instance mismatch')
        else:
            self.dll=ct.CDLL(self.path,mode=os.RTLD_NOLOAD|os.RTLD_LOCAL)
        self.create=self.function('cusparseCreate',(ct.POINTER(P),))
        self.destroy=self.function('cusparseDestroy',(P,))
        self.set_stream=self.function('cusparseSetStream',(P,P))
        self.get_stream=self.function('cusparseGetStream',(P,ct.POINTER(P)))
        self.set_pointer_mode=self.function('cusparseSetPointerMode',(P,I))
        self.version_fn=self.function('cusparseGetVersion',(P,ct.POINTER(I)))
        self.create_descr=self.function('cusparseSpSM_createDescr',(ct.POINTER(P),))
        self.destroy_descr=self.function('cusparseSpSM_destroyDescr',(P,))
        self.buffer_size=self.function('cusparseSpSM_bufferSize',SP_SM_ARGS+(ct.POINTER(ct.c_size_t),))
        self.analysis=self.function('cusparseSpSM_analysis',SP_SM_ARGS+(P,))
        self.solve=self.function('cusparseSpSM_solve',SP_SM_ARGS)
        probe=P();self.check(self.create(ct.byref(probe)))
        try:
            actual=I();self.check(self.version_fn(probe,ct.byref(actual)))
            self.version=actual.value
            if self.version!=expected:raise RuntimeError('CuPy/native cuSPARSE versions differ')
        finally:self.check(self.destroy(probe))
        self.identity=dict(path=self.path,cuda_runtime=self.runtime,cusparse_version=self.version,
            cupy_cusparse_version=expected,solve_argument_count=len(SP_SM_ARGS),same_loaded_library=True)

    def function(self,name,args):
        fn=getattr(self.dll,name);fn.argtypes=list(args);fn.restype=I
        return fn

    @staticmethod
    def check(status):
        if status!=0:raise RuntimeError('Native cuSPARSE status '+str(status))


class NativeTriangle:
    def __init__(self,library,handle,reference):
        self.library=library;self.reference=reference;self.handle=handle
        self.desc=P();self.closed=True
        library.check(library.create_descr(ct.byref(self.desc)));self.closed=False
        self.alpha=reference.alpha  # Persistent host scalar; host pointer mode.
        self.args=(handle,reference.op,reference.op,self.alpha.ctypes.data,
            reference.a.desc,reference.b.desc,reference.c.desc,reference.dtype,reference.algo,self.desc)
        try:
            size=ct.c_size_t();library.check(library.buffer_size(*self.args,ct.byref(size)))
            self.workspace=library.cp.empty(size.value,dtype=library.cp.int8)
            library.check(library.analysis(*self.args,self.workspace.data.ptr))
            self.analysis_calls=1
        except BaseException:
            self.close();raise

    def solve(self):
        if self.closed:raise RuntimeError('Native triangular plan is closed')
        self.reference.out.fill(0.)
        # No CuPy cuSPARSE wrapper, stream mutation or analysis during capture.
        self.library.check(self.library.solve(*self.args))

    def close(self):
        if not self.closed:
            self.library.check(self.library.destroy_descr(self.desc));self.closed=True


class NativeCachedLU:
    """Wrap an existing CachedLU, sharing factors and buffers with its fallback."""
    def __init__(self,fallback,library=None):
        self.fallback=fallback;self.library=library or NativeLibrary();self.cp=fallback.cp
        self.enabled=True;self.closed=False;self.handle=P();self.stream=None
        self.lower=None;self.upper=None;self.solve_calls=0
        self.library.check(self.library.create(ct.byref(self.handle)))
        try:
            self.library.check(self.library.set_pointer_mode(self.handle,0))  # HOST
            self.bind_stream()
            self.lower=NativeTriangle(self.library,self.handle,fallback.lower)
            self.upper=NativeTriangle(self.library,self.handle,fallback.upper)
        except BaseException:
            self.close(close_fallback=False);raise

    def __getattr__(self,name):return getattr(self.fallback,name)

    @property
    def cuda_graph_capture_supported(self):return self.enabled and not self.closed

    @property
    def analysis_calls(self):return self.lower.analysis_calls+self.upper.analysis_calls

    def bind_stream(self,stream=None):
        if self.closed:raise RuntimeError('Native factor is closed')
        stream=stream or self.cp.cuda.get_current_stream()
        if self.cp.cuda.runtime.streamIsCapturing(stream.ptr):
            raise RuntimeError('Bind the native handle before graph capture')
        if self.stream is not None and self.stream.ptr!=stream.ptr:self.stream.synchronize()
        self.library.check(self.library.set_stream(self.handle,stream.ptr));self.stream=stream
        actual=P();self.library.check(self.library.get_stream(self.handle,ct.byref(actual)))
        if (actual.value or 0)!=stream.ptr:raise RuntimeError('Native stream binding mismatch')

    def solve(self):
        if self.closed:raise RuntimeError('Native factor is closed')
        if not self.enabled:return self.fallback.solve()
        if self.cp.cuda.get_current_stream().ptr!=self.stream.ptr:
            raise RuntimeError('Native handle is bound to another stream; bind_stream before capture/execution')
        f=self.fallback;args=(np.int32(f.n),np.int32(f.nrhs));grid=((f.n*f.nrhs+255)//256,)
        f.permute(grid,(256,),(f.rhs,f.perm_r,f.b,*args))
        self.lower.solve();self.upper.solve()
        f.permute(grid,(256,),(f.z,f.perm_c,f.result,*args))
        self.solve_calls+=1
        return f.result

    def close(self,close_fallback=True):
        if not self.closed:
            if self.stream is not None:self.stream.synchronize()
            if self.lower is not None:self.lower.close()
            if self.upper is not None:self.upper.close()
            if self.handle.value:self.library.check(self.library.destroy(self.handle))
            self.closed=True
            if close_fallback:self.fallback.close()

    def bytes(self):
        return self.fallback.bytes()+self.lower.workspace.nbytes+self.upper.workspace.nbytes

    def __del__(self):
        if not getattr(self,'closed',True):
            try:self.close()
            except Exception:pass
