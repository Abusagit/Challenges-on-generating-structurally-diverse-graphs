import numpy as np
import math

from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian


def von_neumann_entropy(density_matrix, cutoff=3, *args, **kwargs):
    
    x = np.mat(density_matrix)
    one = np.identity(x.shape[0])
    base = one - x
    power = base * base
    result = np.trace(base)
    for k in range(2, cutoff):
        result -= np.trace(power) / (k * k - k)
        power = power.dot(base)
    result -= np.trace(power) / (cutoff - 1)
    return result / math.log(2)  # convert from nats to bits


def renyi_entropy(X, q=None, *args, **kwargs):
    """
    Calculate the Rényi entropy with order :math:`q`, or the Von Neumann
    entropy if :math:`q` is `None` or 1.
    """
    # Note that where there are many zero eigenvalues (i.e., large
    # values of beta) in the density matrix, floating-point precision
    # issues mean that there will be negative eigenvalues and the
    # eigenvalues will not sum to precisely one. To avoid encountering
    # `nan`s in `np.log2`, we remove all eigenvalues that are close
    # to zero within 1e-6 tolerance. As for the eigenvalues not summing
    # to exactly one, this is a small source of error in the
    # calculation.
    eigs = np.linalg.eigvalsh(X)
    zero_eigenvalues = np.isclose(np.abs(eigs), 0, atol=1e-6)
    eigs = eigs[np.logical_not(zero_eigenvalues)]

    if q is None or q == 1:
        # plain Von Neumann entropy
        H = -1 * np.sum(eigs * np.log2(eigs))
    else:
        prefactor = 1 / (1 - q)
        H = prefactor * np.log((eigs ** q).sum())
    # print(H)
    return H

def fast_VNGE(A):
    L = laplacian(A)
    # L = np.diag(np.sum(A, axis=1)) - A

    degrees = np.sum(A, axis=1)
    
    C = np.sum(degrees)
    # print(C)
    nonzero_entries = A[np.nonzero(A)]
    
    graph_entropy_taylor = 1 - (np.sum(np.power(degrees, 2)) + 2 * np.sum(np.power(nonzero_entries, 2)))
    # graph_entropy_taylor = 1 - np.power(C, 2) * (degrees @ degrees + 2 * nonzero_entries @ nonzero_entries)
    
    # D = np.diag(degrees)
    
    
    eigenvalue_max = eigsh(L, 1, which='LM', return_eigenvectors=False)[0] # compute largest eigenvalue
    
    
    # breakpoint()
    VNGE = - graph_entropy_taylor * np.log(eigenvalue_max)
    
    # breakpoint()
    
    return VNGE