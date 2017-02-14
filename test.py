#!/usr/bin/env python3

import numpy as np
import numpy.random as npr

from block import block

def test_np():
    npr.seed(0)

    nx, nineq, neq = 4, 6, 7
    Q = npr.randn(nx, nx)
    G = npr.randn(nineq, nx)
    A = npr.randn(neq, nx)
    D = np.diag(npr.rand(nineq))

    K_ = np.bmat((
        (Q, np.zeros((nx,nineq)), G.T, A.T),
        (np.zeros((nineq,nx)), D, np.eye(nineq), np.zeros((nineq,neq))),
        (G, np.eye(nineq), np.zeros((nineq,nineq+neq))),
        (A, np.zeros((neq, nineq+nineq+neq)))
    ))

    K = block((
        (Q,   0, G.T, A.T),
        (0,   D, 'I',   0),
        (G, 'I',   0,   0),
        (A,   0,   0,   0)
    ))

    assert np.allclose(K_, K)

def test_torch():
    try:
        import torch
    except:
        print('Warning: PyTorch not found. Skipping tests.')
        return

    torch.manual_seed(0)

    nx, nineq, neq = 4, 6, 7
    Q = torch.randn(nx, nx)
    G = torch.randn(nineq, nx)
    A = torch.randn(neq, nx)
    D = torch.diag(torch.rand(nineq))


    K_ = torch.cat((
        torch.cat((Q, torch.zeros(nx, nineq).type_as(Q), G.t(), A.t()), 1),
        torch.cat((torch.zeros(nineq, nx).type_as(Q), D, torch.eye(nineq).type_as(Q),
                   torch.zeros(nineq,neq).type_as(Q)), 1),
        torch.cat((G, torch.eye(nineq).type_as(Q), torch.zeros(nineq,nineq+neq).type_as(Q)), 1),
        torch.cat((A, torch.zeros((neq, nineq+nineq+neq))), 1)
    ))

    K = block((
        (Q,   0, G.t(), A.t()),
        (0,   D,   'I',     0),
        (G, 'I',     0,     0),
        (A,   0,     0,     0)
    ))

    assert (K - K_).norm() == 0.0

def test_empty():
    A = npr.randn(3,0)
    B = npr.randn(3,3)
    out = block([[A,B]])
    assert np.linalg.norm(out-B) == 0.0

    A = npr.randn(0,3)
    B = npr.randn(3,3)
    out = block([[A], [B]])
    assert np.linalg.norm(out-B) == 0.0

if __name__=='__main__':
    test_np()
    test_torch()
    test_empty()
