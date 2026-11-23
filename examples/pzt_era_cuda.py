"""Exact FP64 SVD through NVIDIA's generic 64-bit cuSOLVER API.

The legacy CuPy gesvd buffer-size call rejects the full 65534-by-32768 Hankel
problem. Xgesvd uses int64 dimensions and size_t byte workspaces instead.
API: https://docs.nvidia.com/cuda/archive/12.9.1/cusolver/index.html
"""
import ctypes as ct
from importlib.metadata import distribution

import numpy as np


def bind(library):
    pointer, integer, dimension = ct.c_void_p, ct.c_int, ct.c_int64
    prefix = [pointer, pointer, ct.c_byte, ct.c_byte, dimension, dimension,
              integer, pointer, dimension, integer, pointer, integer, pointer,
              dimension, integer, pointer, dimension, integer]
    definitions = {
        'cusolverDnCreateParams': [ct.POINTER(pointer)],
        'cusolverDnDestroyParams': [pointer],
        'cusolverDnSetStream': [pointer, pointer],
        'cusolverDnXgesvd_bufferSize': prefix + [ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)],
        'cusolverDnXgesvd': prefix + [pointer, ct.c_size_t, pointer, ct.c_size_t, pointer],
    }
    for name, arguments in definitions.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, integer
    return library


def exact_svd(matrix, *, event=None):
    import cupy as cp
    import psutil
    if matrix.ndim != 2 or matrix.dtype != cp.float64 or matrix.shape[0] < matrix.shape[1] or matrix.shape[1] < 1:
        raise ValueError('Real FP64 matrix with rows >= columns required')
    # Initializing CuPy's handle first also loads the pinned CUDA dependencies.
    handle = cp.cuda.device.get_cusolver_handle()
    path = distribution('nvidia-cusolver-cu12').locate_file('nvidia/cusolver/lib/libcusolver.so.11')
    library = bind(ct.CDLL(str(path)))
    params = ct.c_void_p()
    def check(status, operation):
        if status:
            raise RuntimeError(f'{operation} returned cuSOLVER status {status}')
    check(library.cusolverDnCreateParams(ct.byref(params)), 'cusolverDnCreateParams')
    try:
        stream = cp.cuda.get_current_stream()
        check(library.cusolverDnSetStream(handle, stream.ptr), 'cusolverDnSetStream')
        m, n = matrix.shape
        a = cp.array(matrix, dtype=cp.float64, order='F', copy=True)
        u = cp.empty((m, n), dtype=cp.float64, order='F')
        singular = cp.empty(n, dtype=cp.float64)
        vt = cp.empty((n, n), dtype=cp.float64, order='F')
        info = cp.empty(1, dtype=cp.int32)
        # CUDA_R_64F = 1 in NVIDIA's library_types.h. 'S' requests economy vectors.
        arguments = [handle, params, ord('S'), ord('S'), m, n,
                     1, a.data.ptr, m, 1, singular.data.ptr,
                     1, u.data.ptr, m, 1, vt.data.ptr, n, 1]
        device_bytes, host_bytes = ct.c_size_t(), ct.c_size_t()
        check(library.cusolverDnXgesvd_bufferSize(*arguments, ct.byref(device_bytes), ct.byref(host_bytes)),
              'cusolverDnXgesvd_bufferSize')
        free, _ = cp.cuda.runtime.memGetInfo()
        # The matrix and output vectors are already counted in memGetInfo.
        if device_bytes.value > free or host_bytes.value > psutil.virtual_memory().available:
            raise MemoryError('Reported Xgesvd workspace exceeds available memory; accuracy is not reduced')
        if event:
            event('exact_64bit_svd_workspace', device_workspace_bytes=device_bytes.value,
                  host_workspace_bytes=host_bytes.value, free_device_bytes=free)
        device_work = cp.cuda.alloc(max(1, device_bytes.value))
        host_work = np.empty(max(1, host_bytes.value), dtype=np.uint8)
        check(library.cusolverDnXgesvd(*arguments, device_work.ptr, device_bytes.value,
              host_work.ctypes.data, host_bytes.value, info.data.ptr), 'cusolverDnXgesvd')
        stream.synchronize()
        result = int(cp.asnumpy(info)[0])
        if result:
            raise ArithmeticError(f'Exact 64-bit SVD failed to converge or had invalid parameter: info={result}')
        return u, singular, vt
    finally:
        library.cusolverDnDestroyParams(params)
