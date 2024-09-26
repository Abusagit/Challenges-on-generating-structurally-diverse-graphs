import json
import pickle
import time
import logging
from functools import total_ordering
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, MutableSet


import igraph as ig
import networkx as nx
import numpy as np

from distances import precompute_assets_for_generated_graphs

Graph_representation = Tuple[str, Any]
Distance_func = Any
Distance_name = str
Graph_random_model = str
Metainfo = Any
Graph = nx.classes.graph.Graph
Parameters = Dict[str, Any]

logger : logging.Logger = logging.getLogger(__name__)

VOLUME_METRICS_IMPLEMENTED = {"min", "avg", "energy"}


def check_for_implementation(volume_func_type: str):

    if volume_func_type not in VOLUME_METRICS_IMPLEMENTED:
        raise NotImplementedError(f"""Volume func type '{volume_func_type}' is not in the list of supported at current stage of development. 
                                        Available options are: {list(VOLUME_METRICS_IMPLEMENTED)}.
                                    """)
        
class ArgumentError(BaseException):
    pass



class Config:
    """
    Convinient interface for storing graphs generation parameters from .json file
    
    """
    
    def __init__(self, config_file: Path) -> None:
        self.config_file = config_file
        self.settings = read_json(self.config_file)
        
        
        self.models_configurations = self.settings["models"]
        self.distances_configurations = self.settings["distances"]
        self.nodes_number = self.settings["nodes"]
        self.equal_sizes = self.settings["equal_sample_size_per_model"]
        self.samples = self.settings["samples"]    
        # self.model_common_params = settings["models_common"]
        self.orca_path = Path(self.settings["orca_path"])
        
        
        self.greedy_models = self.settings["models"]
        # self.greedy_models_fractions = self.settings["greedy"]["fractions"]
        self.greedy_distances = self.settings["distances"]
        self.greedy_sampling_size = self.settings["greedy"]["sampling_size"]        
        self.greedy_initial_graphs = self.settings["greedy"]["initial_graphs"]
        
                
        self.genetic_config = self.settings["genetic"]
        
        try:
            self.genetic_checkpoint_interval = self.genetic_config["checkpoints_interval"]
            del self.genetic_config["checkpoints_interval"]
            
        except KeyError:
            self.genetic_checkpoint_interval = 1000
            
        self.localopt_config = self.settings.get("localopt")
        
        
class GraphModel:
    """
    class for storing generated graphs for given model with <label>
    """
    
    def __init__(self, 
                 label: str, 
                 graphs: List[np.ndarray], 
                 precomputed_descriptors: List[Any]=None,
                ) -> None:
        
        """
        label: label for given graph model (e.g. "ER_p_0_5")
        graphs: list of generated graphs
        precomputed descriptors: 
        """
        
        self.label = label
        self.graphs = graphs
        self.precomputed_descriptors = precomputed_descriptors  # graph-wise and for every metric which can be precomputed
    
    def __repr__(self) -> str:
        string = "Graph model with parameters: {}\n{} graphs\nPrecomputed descriptors: {}\n"
        
        return string.format(self.label, len(self.graphs), ', '.join(self.precomputed_descriptors.keys()))


class DistanceReport:
    
    """class for storing report for <distance_lanel>"""
    
    def __init__(self, distance_label: str, 
                 model_combination_C_n_k_to_volume: Dict[int, Any], 
                 model_combinations_with_max_volumes: Dict[int, int]) -> None:
        
        self.label = distance_label
        self.report_dict = model_combination_C_n_k_to_volume
        self.model_combinations_with_max_volumes = model_combinations_with_max_volumes
        
    def __repr__(self) -> str:
        
        string = "Report of {}:\nreport_dict:\n{}\nmodel_combinations_with_max_volumes:\n{}\n".format(self.label, self.report_dict, self.model_combinations_with_max_volumes)
        
        return string
    
    
    

Graphs_dict = Dict[Distance_name, Dict[str, GraphModel]]
Report = Any    

def get_label(prefix, items):
    postfix = '_'.join(f"{k}_{v}" for k, v in sorted(items.items()))
    if postfix:
        return '_'.join([prefix, postfix])    
    
    return prefix

def read_pickle(filename: Path):
    with open(filename, "rb") as handler:
        obj = pickle.load(handler)
    return obj

        
def save_npy(array, to_filename, dtype=np.float64):
    with open(to_filename, mode="wb") as handler:
        np.save(handler, array, dtype=dtype) # type: ignore
        
def save_pickle(obj, to_filename):
    with open(to_filename, mode="wb") as handler:
        pickle.dump(obj, handler)
        
def read_json(filename):
    with open(filename) as handler:
        return json.load(handler)

def save_json(obj, filename):
    with open(filename, "w") as handler:
        handler.write(json.dumps(obj=obj, indent=4))

def nx_to_igraph(nx_graph:Graph):
    return ig.Graph.Adjacency((nx.to_numpy_matrix(nx_graph) > 0).tolist()) # type: ignore

def igraph_to_nx(ig_graph: ig.Graph):
    return ig_graph.to_networkx()


def time_wrapper(func):
    
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        
        logger.info(f"Function {func.__name__}, took {t2 - t1:.3f} seconds to run")
        
        return result
    
    return wrapper


def transform_generated_graphs_dict_to_list(
              generated_graphs: List[Graph] or List[Tuple[np.ndarray, str, Any]],
              orca_path: str,
              njobs:int,
              distances_set: MutableSet[Distance_name]
    ) -> Dict[Distance_func, List[GraphModel]]:
    
    if isinstance(generated_graphs[0], (tuple, list)):
        _assets, graphs = [], []
        
        for g, _, asset in generated_graphs:
            _assets.append(asset)
            graphs.append(g)
        assets = {}
        for distance in distances_set:
            assets[distance] = _assets
    else:
        
        graphs = generated_graphs
        assets: Dict[Distance_name, Any] = precompute_assets_for_generated_graphs(generated_graphs=generated_graphs,
                                                        distances_names_to_precompute_assets=distances_set,
                                                        orca_path=orca_path,
                                                        njobs=njobs,
                                                        )
    graph_models = [
            GraphModel(label="Generated", graphs=graphs, precomputed_descriptors=assets)
    ]
    
    return graph_models

def get_graphs_from_metareport(metareport: Dict[str, List[Any]]):
    """
    Extract graphs from `metareport`
    """
    return list(map(lambda x: x.graph, metareport["population"]))

def load_pickle_run_by_index(directory, index) -> Dict[str, Any]:
    """
    Loads metareport with particular `index` from `directory` 
    """
    return pickle.load(file=open(directory / "metareports_chunks" / f"report_{index}.pkl", "rb"))

def extract_graphs_from_run_by_index(directory_root_path: Path, index: int):
    """
    Extract graphs from metareport with `index`
    """
    return get_graphs_from_metareport(load_pickle_run_by_index(directory_root_path, index))



def save_graphs_in_adj_matrices(graphs: List[Graph], save_dir: Union[Path, str], save_name: str) -> None:
    """
    Saves graphs as array of adjacency matrices in .npy file
    """
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    save_filename = save_dir / save_name
    
    graph_arrays = np.array([nx.to_numpy_array(g) for g in graphs])
    np.save(save_filename, graph_arrays)