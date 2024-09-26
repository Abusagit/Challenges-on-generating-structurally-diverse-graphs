import logging
import numpy as np
import netlsd
import networkx as nx
import warnings

from pathlib import Path
from joblib import Parallel, delayed
from scipy.integrate import quad, IntegrationWarning
from functools import partial
from scipy.sparse.csgraph import laplacian
from scipy.linalg import expm
from typing import List, Set, Any, Tuple
from netrd.distance.portrait_divergence import portrait_divergence
from tqdm.auto import tqdm

from graphlets import get_gcm
from entropies import renyi_entropy, von_neumann_entropy



logger = logging.getLogger(__name__)

class ProgressParallel(Parallel):
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
        
        
def GCD(graphlet_corr_matrix_1: np.ndarray, graphlet_correlation_matrix_2: np.ndarray) -> float:
    
    """Computes Graphlet correltation distance (GCD) between 2 graphs
    graphlet_corr_matrix_1: matrix 1
    graphlet_correlation_matrix_2: matrix 2
    
    returns GCD
    """
    
    # euclidian distance of upper triangular of GCMs:
    
    u_t_1 = graphlet_corr_matrix_1[np.triu_indices_from(graphlet_corr_matrix_1, 1)]
    u_t_2 = graphlet_correlation_matrix_2[np.triu_indices_from(graphlet_correlation_matrix_2, 1)]
    
    return np.linalg.norm(u_t_1 - u_t_2)

def im_distance(density_1, density_2) -> float:
    """Computes IM-distance of 2 graphs
    density_1: density function of graph 1
    density_2: density function of graph 2
    
    returns L_2 integral norm of 2 densities
    """
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=IntegrationWarning)
        func = lambda w: (density_1(w) - density_2(w)) ** 2
        res = np.sqrt(quad(func, 0, np.inf, limit=100)[0])
    
    return res


def quantum_JSD(rho_H_1: Tuple[np.ndarray, int], 
                rho_H_2: Tuple[np.ndarray, int], 
                entropy_function: Any, **kwargs) -> float:
    """
    
    computes quantumJSD distance between graph_1 and graph_2. Implements 2 distinct functions - approximate and exact JSD, depends on entropy_functio
    
    rho_H_1: density matrix and entropy of graph_1
    rho_H_2: -//- of graph_2
    entropy_function: function dedicated to entripy computation:
        approximate function => returns approximate JSD, faster
        exact function => returns exact JSD, slower    
    
    available kwargs:
    q - for Renyi Entropy
    cutoff - for Taylor approximation of Von Neumann Graph Entropy
    
    """
    
    rho_1, H_1 = rho_H_1
    rho_2, H_2 = rho_H_2
    
    mix = (rho_1 + rho_2) / 2
    
    H_0 = entropy_function(mix, **kwargs)
    
    distance = np.sqrt(np.abs(H_0 - 0.5 * (H_1 + H_2)))
    
    return distance


def IM_density(w, norm, W, hwhm) -> float:
    return np.sum(hwhm / ((w - W) ** 2 + hwhm ** 2)) / norm


def density_matrix(A, beta=0.1):
    
    
    """
    Create the density matrix encoding probabilities for entropies.
    This is done using a fictive diffusion process with time parameter
    :math:`beta`.
    """
    
    if beta <= 0:
            raise ValueError("beta must be positive.")
    
    if isinstance(A, nx.classes.graph.Graph):
        A = nx.to_numpy_array(A)
    
    L = laplacian(A)
    rho = expm(-1 * beta * L)
    rho = rho / np.trace(rho)

    return rho


##### Precomputation part contains functions for precomputation of objects needed for various graph distances

def portrait(G):
    """
    Parameters
    ----------
    G (nx.Graph or nx.DiGraph):
        a graph.

    Returns
    -------
    B (np.ndarray):
        a matrix :math:`B` such that :math:`B_{i,j}` is the number of starting
        nodes in graph with :math:`j` nodes in shell :math:`i`.
    """
    
    if not isinstance(G, nx.classes.Graph):
        G = nx.from_numpy_array(G)
    
    N = G.number_of_nodes()
    try:
        dia = nx.diameter(G)
    except nx.exception.NetworkXError:
        dia = N # graph is disconnected

    # B indices are 0...dia x 0...N-1:
    B = np.zeros((dia + 1, N))

    max_path = 1
    adj = G.adj

    for starting_node in G.nodes():
        nodes_visited = {starting_node: 0}
        search_queue = [starting_node]
        d = 1

        while search_queue:
            next_depth = []
            extend = next_depth.extend

            for n in search_queue:
                l = [i for i in adj[n] if i not in nodes_visited]
                extend(l)

                for j in l:
                    nodes_visited[j] = d

            search_queue = next_depth
            d += 1

        node_distances = nodes_visited.values()
        max_node_distances = max(node_distances)

        curr_max_path = max_node_distances
        if curr_max_path > max_path:
            max_path = curr_max_path

        # build individual distribution:
        dict_distribution = dict.fromkeys(node_distances, 0)
        for d in node_distances:
            dict_distribution[d] += 1

        # add individual distribution to matrix:
        for shell, count in dict_distribution.items():
            B[shell][count] += 1

        # HACK: count starting nodes that have zero nodes in farther shells
        max_shell = dia
        while max_shell > max_node_distances:
            B[max_shell][0] += 1
            max_shell -= 1

    return B[: max_path + 1, :]

def precompute_IM_asset(graph: nx.classes.graph.Graph, hwhm=0.08):
    
    if not isinstance(graph, np.ndarray):
        adj = nx.to_numpy_array(graph)
    else:
        adj = graph
        
    N = len(adj)
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        L = laplacian(adj, normed=False)
    
    # get the modes for the positive-semidefinite laplacian
    
    W = np.sqrt(np.abs(np.linalg.eigh(L)[0][1:]))
    
    # calculate the norm of spectrum
    norm = (N - 1) * np.pi / 2 - np.sum(np.arctan(-W / hwhm))
    
    return partial(IM_density, norm=norm, W=W, hwhm=hwhm)

WARNING_MESSAGE = "JSD is only a metric for 0 ≤ q < 2"

def precompute_graph_density_renyi_entropy(density_matrix, q=None):

    if q and q >= 2:
        logger.warning(WARNING_MESSAGE)
        warnings.warn(WARNING_MESSAGE, RuntimeWarning)
    
    H = renyi_entropy(density_matrix, q)
    return density_matrix, H

def precompute_graph_density_taylor_approx_entropy(density_matrix, cutoff=3):
    
    H = von_neumann_entropy(density_matrix=density_matrix, cutoff=cutoff)
    
    return density_matrix, H
    
def precompute_graphlet_correlation_matrix(graph: nx.classes.graph.Graph, orca_path:Path):
    if isinstance(graph, np.ndarray):
        graph = nx.from_numpy_array(graph)
        
    return get_gcm(edge_list=graph.edges(), 
                   orca_prefix=orca_path,
                   nodes_num=len(graph.nodes()),
                   )

def compute_assets_in_parallel(parallel_workers: Parallel, asset_function: Any, graphs_entities: List[Any], **kwargs):
    
    return list(parallel_workers(delayed(function=asset_function)(asset, **kwargs) for asset in graphs_entities))
    

def precompute_assets_for_generated_graphs(generated_graphs: List[nx.classes.graph.Graph],
                                           distances_names_to_precompute_assets: Set[str],
                                           orca_path:Path,
                                           njobs:int,
                                           backend:str="multiprocessing", 
                                           bath_size:str="auto",
                                           disable_tqdm=False,
                                           ):
    
    with ProgressParallel(n_jobs=njobs, backend=backend, batch_size=bath_size) as parallel_workers:
    
        compute_assets = partial(compute_assets_in_parallel, parallel_workers=parallel_workers)
        
        REQUIRES_DENSITY_MATRIX = {"quantumJSD", "quantumJSD_approx"}
        
        precomputed_assets_dict = {}

        if REQUIRES_DENSITY_MATRIX & distances_names_to_precompute_assets:
            # compute graph_densities_first
            density_matrices = compute_assets(asset_function=density_matrix, 
                                            graphs_entities=generated_graphs)
            
            # compute assets requiring density matrix:
            for distance_name in distances_names_to_precompute_assets & REQUIRES_DENSITY_MATRIX:
                precomputed_assets_dict[distance_name] = compute_assets(asset_function=PRECOMPUTE_DICT[distance_name],
                                                                        graphs_entities=density_matrices)
            
        
        
        # compute assets requiring nx Graph instance:
        
        for distance_name in distances_names_to_precompute_assets - REQUIRES_DENSITY_MATRIX - {"GCD"}:
            precomputed_assets_dict[distance_name] = compute_assets(asset_function=PRECOMPUTE_DICT[distance_name],
                                                                    graphs_entities=generated_graphs)
        
        if "GCD" in distances_names_to_precompute_assets:
            precomputed_assets_dict["GCD"] = compute_assets(asset_function=PRECOMPUTE_DICT["GCD"],
                                                            graphs_entities=generated_graphs,
                                                            orca_path=orca_path)
    
    return precomputed_assets_dict
    
  
    

PRECOMPUTE_DICT = {
    "IM-distance": precompute_IM_asset,
    "netLSD_wave": netlsd.wave,
    "netLSD_heat": netlsd.heat,
    "GCD": precompute_graphlet_correlation_matrix, 
    "Portrait": portrait,
    "IM-distance": precompute_IM_asset,
    "quantumJSD_approx": precompute_graph_density_taylor_approx_entropy,
    "quantumJSD": precompute_graph_density_renyi_entropy,
}

DISTANCE_TO_FUNC = {
    # mapper from distance name to distance function
    "netLSD_wave": netlsd.compare,
    "netLSD_heat": netlsd.compare,
    "GCD": GCD, 
    "Portrait": portrait_divergence,
    "IM-distance": im_distance,
    "quantumJSD_approx": partial(quantum_JSD, entropy_function=von_neumann_entropy),
    "quantumJSD": partial(quantum_JSD, entropy_function=renyi_entropy),
}
