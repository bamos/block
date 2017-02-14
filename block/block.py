import numpy as np

try:
    import torch
except:
    pass


import re
from abc import ABCMeta, abstractmethod


def block(rows, dtype=None, arrtype=None):
    if (not _is_list_or_tup(rows)) or len(rows) == 0 or \
       np.any([not _is_list_or_tup(row) for row in rows]):
        raise RuntimeError('''
Unexpected input: Expected a non-empty list of lists.
If you are interested in helping expand the functionality
for your use case please send in an issue or PR at
http://github.com/bamos/block''')

    rowLens = [len(row) for row in rows]
    if len(np.unique(rowLens)) > 1:
        raise RuntimeError('''
Unexpected input: Rows are not the same length.
Row lengths: {}'''.format(rowLens))

    nCols = len(rows)
    nRows = rowLens[0]
    rowSizes = np.zeros(nRows, dtype=int)
    colSizes = np.zeros(nCols, dtype=int)

    backend = _get_backend(rows, dtype, arrtype)

    for i, row in enumerate(rows):
        for j, elem in enumerate(row):
            if backend.is_complete(elem):
                rowSz, colSz = backend.extract_shape(elem)
                rowSizes[i] = rowSz
                colSizes[j] = colSz

    cRows = []
    for row, rowSz in zip(rows, rowSizes):
        rowSz = int(rowSz)
        cCol = []
        for elem, colSz in zip(row, colSizes):
            colSz = int(colSz)
            # TODO: Check types.
            if backend.is_complete(elem):
                cElem = elem
            elif isinstance(elem, float) or isinstance(elem, int):
                cElem = backend.build_full((rowSz, colSz), elem)
            elif isinstance(elem, str):
                if elem == 'I':
                    assert(rowSz == colSz)
                    cElem = backend.build_eye(rowSz)
                elif elem == '-I':
                    assert(rowSz == colSz)
                    cElem = -backend.build_eye(rowSz)
                else:
                    assert(False)
            else:
                assert(False)
            cCol.append(cElem)
        cRows.append(cCol)

    return backend.build(cRows)



def _is_list_or_tup(x):
    return isinstance(x, list) or isinstance(x, tuple)


def _get_backend(rows, dtype, arrtype):
    if arrtype == np.ndarray and dtype is not None:
        return NumpyBackend(arrtype, dtype)
    elif arrtype is not None and re.search('torch\..*Tensor', arrtype):
        return TorchBackend(arrtype, dtype)
    else:
        npb = NumpyBackend()
        tb = TorchBackend()
        for row in rows:
            for elem in row:
                if npb.is_complete(elem):
                    if dtype is None:
                        dtype = type(elem[0, 0])
                    if arrtype is None:
                        arrtype = type(elem)
                    return NumpyBackend(dtype, arrtype)
                elif tb.is_complete(elem):
                    return TorchBackend(type(elem))

    assert(False)


class Backend():
    __metaclass__=ABCMeta

    @abstractmethod
    def extract_shape(self, x): pass

    @abstractmethod
    def build_eye(self, n): pass

    @abstractmethod
    def build_full(self, shape, fill_val): pass

    @abstractmethod
    def build(self, rows): pass

    @abstractmethod
    def is_complete(self, rows): pass


class NumpyBackend(Backend):

    def __init__(self, dtype=None, arrtype=None):
        self.dtype = dtype
        self.arrtype = arrtype

    def extract_shape(self, x):
        return x.shape

    def build_eye(self, n):
        return np.eye(n)

    def build_full(self, shape, fill_val):
        return np.full(shape, fill_val, self.dtype)

    def build(self, rows):
        return np.bmat(rows)

    def is_complete(self, x):
        return isinstance(x, np.ndarray)


class TorchBackend(Backend):

    def __init__(self, dtype=None):
        self.dtype = dtype

    def extract_shape(self, x):
        return x.size()

    def build_eye(self, n):
        return torch.eye(n).type(self.dtype)

    def build_full(self, shape, fill_val):
        return fill_val * torch.ones(*shape).type(self.dtype)

    def build(self, rows):
        compRows = []
        for row in rows:
            compRows.append(torch.cat(row, 1))
        return torch.cat(compRows)

    def is_complete(self, x):
        return re.search('torch\..*Tensor', str(x.__class__))
