__doc__ = """
    Dedicated module for graphlet precomputation. Developed for using only while generating, but supports calls
    from newly created functions.
"""

import os
from subprocess import Popen, PIPE
import tempfile
import pandas as pd
import numpy as np
from scipy.stats import rankdata


def run_orca(edge_list, nodes_num, orca_prefix='', graphlet_size=4):
    
    """
    Python wrapper for c++ ORCA start
    
    """
    
    command = os.path.join(orca_prefix, 'orca')
    
    # Create temporary files
    fd1, in_file = tempfile.mkstemp()
    fd2, out_file = tempfile.mkstemp()

    # Fill input file
    with open(in_file, 'w') as fp:
        edges_num = len(edge_list)
        
        fp.write(f"{nodes_num} {edges_num}\n")
        
        edges_in_string = '\n'.join(f"{source} {target}" for source, target in edge_list)
        fp.write(f"{edges_in_string}\n")

    # Run orca
    full_args = [command] + ["node"] + ['%d' % graphlet_size, in_file, out_file]
    FNULL = open(os.devnull, 'w')
    
    proc = Popen(full_args, bufsize=0, stdin=None, stdout=FNULL,
                 stderr=PIPE, universal_newlines=True)
    err_msg = proc.stderr.read().strip()
    proc.stderr.close()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(err_msg)
    
    # Read results
    G = pd.read_table(out_file, header=None, sep=' ')
    # G.index = used_labels

    os.close(fd1)
    os.close(fd2)
    os.remove(in_file)
    os.remove(out_file)

    return G


def _normalize(X):
    Xn = np.subtract(X, X.mean(1)[:, np.newaxis])
    np.divide(Xn, np.sqrt(np.sum(Xn * Xn, 1) + 1e-8)[:, np.newaxis], Xn)
    return Xn


def pearson(X, Y):
    """
    All-against-all Pearson correlations for data matrices X and Y
    """
    Xn = _normalize(X)
    Yn = _normalize(Y)
    R = np.dot(Xn, Yn.T)
    R = np.maximum(np.minimum(R, 1.0, R), -1.0)
    return R


def spearman(X, Y):
    """
    All-against-all Spearman correlations for data matrices X and Y
    """
    Xr = np.apply_along_axis(rankdata, 1, X)
    Yr = np.apply_along_axis(rankdata, 1, Y)
    return pearson(Xr, Yr)


def get_gcm(edge_list, nodes_num, orca_prefix='', graphlet_size=4):
    GCM_ORDER = [0, 2, 5, 7, 8, 10, 11, 6, 9, 4, 1]
    
    GDC = run_orca(edge_list, 
                   orca_prefix=orca_prefix, 
                   graphlet_size=graphlet_size,
                   nodes_num=nodes_num
                )    

    # Calculate correlations between orbits
    GDC1 = GDC.iloc[:, GCM_ORDER].values.T
    GCM = spearman(GDC1, GDC1)
    GCM[np.isnan(GCM)] = 0.0
    
    return GCM