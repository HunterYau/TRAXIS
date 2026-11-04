"""Bounded serial Elmer stiffness/consistent-mass imports without dense expansion."""

from pathlib import Path

import numpy as np
from scipy import sparse

from traxis.storage import atomic_write


def read_elmer_sparse_operators(directory: Path, nodes: int, *, max_entries=4000000, max_file_mib=256):
    if type(nodes) is not int or not 4 <= nodes <= 20000:
        raise ValueError('sparse operator import is limited to 4–20000 nodes')
    if type(max_entries) is not int or not 1 <= max_entries <= 4000000 or not 0 < max_file_mib <= 256:
        raise ValueError('invalid sparse import bounds')
    directory = Path(directory)
    if (directory / 'operator_sizes.dat').stat().st_size > 1024:
        raise MemoryError('operator dimensions exceed 1 KiB')
    sizes = np.loadtxt(directory / 'operator_sizes.dat', ndmin=1)
    ndof = nodes * 3
    if (sizes.shape != (5,) or not np.isfinite(sizes).all() or not np.array_equal(sizes, sizes.astype(np.int64))
            or sizes[0] != ndof or sizes[2] != nodes or not 1 <= sizes[1] <= max_entries
            or sizes[3] != 1 or sizes[4] != ndof):
        raise ValueError('invalid/oversized serial operator dimensions')
    if (directory / 'operator_a.dat').stat().st_size > max_file_mib * 1024**2:
        raise MemoryError('sparse operator triplets exceed the file limit')
    if (directory / 'operator_perm.dat').stat().st_size > 1024**2:
        raise MemoryError('node permutation exceeds 1 MiB')
    entries = np.loadtxt(directory / 'operator_a.dat', ndmin=2)
    permutation = np.loadtxt(directory / 'operator_perm.dat', ndmin=2)
    if (entries.shape != (int(sizes[1]), 4) or not np.isfinite(entries).all()
            or permutation.shape != (nodes, 2) or not np.isfinite(permutation).all()
            or not np.array_equal(permutation, permutation.astype(np.int64))
            or not np.array_equal(permutation[:, 0], np.arange(1, nodes + 1))
            or not np.array_equal(np.sort(permutation[:, 1]), np.arange(1, nodes + 1))):
        raise ValueError('invalid sparse values or nonbijective node permutation')
    indexes = entries[:, :2]
    if not np.array_equal(indexes, indexes.astype(np.int64)) or indexes.min() < 1 or indexes.max() > ndof:
        raise ValueError('invalid sparse operator indexes')
    indexes = indexes.astype(np.int64) - 1
    if len(np.unique(indexes[:, 0] * ndof + indexes[:, 1])) != len(indexes):
        raise ValueError('duplicate sparse operator entry')
    order = (3 * (permutation[:, 1].astype(np.int64) - 1)[:, None] + np.arange(3)).ravel()
    matrices = []
    for column in (2, 3):
        matrix = sparse.coo_matrix((entries[:, column], indexes.T), shape=(ndof, ndof)).tocsr()
        matrix = matrix[order][:, order].tocsr()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        matrices.append(matrix)
    return tuple(matrices)


def save_operators(path: Path, stiffness, mass):
    arrays = {'shape': np.array(stiffness.shape)}
    for name, matrix in (('stiffness', stiffness), ('mass', mass)):
        matrix = sparse.csr_matrix(matrix)
        if matrix.shape != stiffness.shape or matrix.shape[0] != matrix.shape[1]:
            raise ValueError('operator dimensions differ')
        for key in ('data', 'indices', 'indptr'):
            arrays[f'{name}_{key}'] = getattr(matrix, key)
    atomic_write(Path(path), lambda stream: np.savez_compressed(stream, **arrays))


def load_operators(path: Path):
    import zipfile
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) != 7 or sum(row.file_size for row in archive.infolist()) > 256 * 1024**2:
            raise MemoryError('operator archive has invalid size/entry count')
    with np.load(path, allow_pickle=False) as saved:
        shape = saved['shape']
        if (shape.shape != (2,) or shape.dtype.kind not in 'iu' or shape[0] != shape[1]
                or not 12 <= shape[0] <= 60000):
            raise ValueError('invalid operator archive dimensions')
        result = []
        for name in ('stiffness', 'mass'):
            data, indices, pointers = [saved[f'{name}_{key}'].copy() for key in ('data', 'indices', 'indptr')]
            if (data.ndim != 1 or indices.shape != data.shape or len(data) > 4000000
                    or not np.isfinite(data).all() or indices.dtype.kind not in 'iu' or pointers.dtype.kind not in 'iu'
                    or pointers.shape != (int(shape[0]) + 1,) or pointers[0] != 0 or pointers[-1] != len(data)
                    or (pointers[1:] < pointers[:-1]).any() or (pointers > len(data)).any()
                    or (indices < 0).any() or (indices >= shape[0]).any()):
                raise ValueError('invalid sparse archive arrays')
            matrix = sparse.csr_matrix((data, indices, pointers), shape=tuple(shape))
            if not matrix.has_canonical_format:
                raise ValueError('sparse archive contains duplicate/unsorted entries')
            result.append(matrix)
    return tuple(result)
