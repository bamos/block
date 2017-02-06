import numpy as np
import numpy.random as npr

try:
    import torch
except:
    pass


import re
from abc import ABCMeta, abstractmethod

def isListOrTup(x):
    return isinstance(x, list) or isinstance(x, tuple)

def block(rows, dtype=None, arrtype=None):
    if (not isListOrTup(rows)) or len(rows) == 0 or \
       np.any([not isListOrTup(row) for row in rows]):
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

    backend = getBackend(rows, dtype, arrtype)

    for i, row in enumerate(rows):
        for j, elem in enumerate(row):
            if backend.isComplete(elem):
                rowSz, colSz = backend.getShape(elem)
                rowSizes[i] = rowSz
                colSizes[j] = colSz

    cRows = []
    for row, rowSz in zip(rows, rowSizes):
        rowSz = int(rowSz)
        cCol = []
        for elem, colSz in zip(row, colSizes):
            colSz = int(colSz)
            # TODO: Check types.
            if backend.isComplete(elem):
                cElem = elem
            elif isinstance(elem, float) or isinstance(elem, int):
                cElem = backend.getFull((rowSz, colSz), elem)
            elif isinstance(elem, str):
                if elem == 'I':
                    assert(rowSz == colSz)
                    cElem = backend.getEye(rowSz)
                else:
                    assert(False)
            else:
                assert(False)
            cCol.append(cElem)
        cRows.append(cCol)

    return backend.build(cRows)

def getBackend(rows, dtype, arrtype):
    if arrtype == np.ndarray and dtype is not None:
        return NumpyBackend(arrtype, dtype)
    elif arrtype is not None and re.search('torch\..*Tensor', arrtype):
        return TorchBackend(arrtype, dtype)
    else:
        npb = NumpyBackend()
        tb = TorchBackend()
        for row in rows:
            for elem in row:
                if npb.isComplete(elem):
                    if dtype is None:
                        dtype = type(elem[0,0])
                    if arrtype is None:
                        arrtype = type(elem)
                    return NumpyBackend(dtype, arrtype)
                elif tb.isComplete(elem):
                    return TorchBackend(type(elem))

    assert(False)

class Backend(metaclass=ABCMeta):
    @abstractmethod
    def getShape(self, x): pass

    @abstractmethod
    def getEye(self, n): pass

    @abstractmethod
    def getFull(self, shape, fill_val): pass

    @abstractmethod
    def build(self, rows): pass

    @abstractmethod
    def isComplete(self, rows): pass

class NumpyBackend(Backend):
    def __init__(self, dtype=None, arrtype=None):
        self.dtype = dtype
        self.arrtype = arrtype

    def getShape(self, x):
        return x.shape

    def getEye(self, n):
        return np.eye(n)

    def getFull(self, shape, fill_val):
        return np.full(shape, fill_val, self.dtype)

    def build(self, rows):
        return np.bmat(rows)

    def isComplete(self, x):
        return isinstance(x, np.ndarray)

class TorchBackend(Backend):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def getShape(self, x):
        return x.size()

    def getEye(self, n):
        return torch.eye(n).type(self.dtype)

    def getFull(self, shape, fill_val):
        return fill_val*torch.ones(*shape).type(self.dtype)

    def build(self, rows):
        compRows = []
        for row in rows:
            compRows.append(torch.cat(row, 1))
        return torch.cat(compRows)

    def isComplete(self, x):
        return re.search('torch\..*Tensor', str(x.__class__))
