import igraph as ig
import networkx as nx
import logging

import os

from joblib import Parallel, delayed
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import numpy as np
from utils import igraph_to_nx, get_label, time_wrapper, GraphModel, ArgumentError, transform_generated_graphs_dict_to_list
from utils import Graph_random_model, Distance_name, Parameters, Graphs_dict
from distances import precompute_assets_for_generated_graphs

from tempfile import TemporaryDirectory

logger = logging.getLogger(__name__)

def generate_powerlaw_graph(n:int, gamma:float) -> nx.classes.graph.Graph:
    number_of_vertices = n
    powerlaw_degree_sequence: List[float] = nx.utils.powerlaw_sequence(n=number_of_vertices, exponent=gamma)
    G: nx.classes.graph.Graph = nx.expected_degree_graph(w=powerlaw_degree_sequence, selfloops=False)
    return G

MODEL_TO_FUNC = {
    # mapper from random model name to generating function
    "ER": ig.Graph.Erdos_Renyi,
    "PA": ig.Graph.Barabasi,
    "SBM": ig.Graph.SBM,
    "WS": nx.random_graphs.watts_strogatz_graph,
    "RGG_ig": ig.Graph.GRG,
    "RGG_nx": nx.random_geometric_graph,
    "Regular": nx.random_graphs.random_regular_graph,
    "PowerlawCluster": nx.random_graphs.powerlaw_cluster_graph,
    "PowerlawSequence": generate_powerlaw_graph,   
}


@time_wrapper
def generate_graphs(models_and_configurations: Dict[Graph_random_model, List[Parameters]], 
                    number_of_nodes: int, 
                    number_of_samples: int, 
                    njobs: int,
                    orca_path:Path,
                    distances_names_set: Set[Distance_name],
                    equal_sizes: bool=True,
                    tmpdir:Optional[TemporaryDirectory]=None,
                    ) -> Graphs_dict:
    """
    generates graphs from given models configurations
    
    models_and_configurations: dict of random graph models and corresponding list of dicts of parameters
    number_of_nodes: number of nodes to be generated in each graph
    number_of_samples: number of samples for each graph model
    njobs: number of threads
    orca_path: path to ORCA package for Graphlet Correlation Distance
    distances_names_set: set of names of distances names
    equal_sizes: whether or not generate full size of graphs for each graph random model. 
            If True, for each random model configuration will be generated `number_of_graphs` // M graphs, where M is the number of configurations for given random model
            If False, for each random model configuration will be generated exactly `number_of_graphs` graphs
    
    tmpdir: if is not None, all adjacencies should be stored there
    
    returns dictionary of distances names and corresponding GraphModels
    """
    
    graphs = defaultdict(dict)
    number_of_models = sum(map(len, list(models_and_configurations.values())))
    
    
    number_of_graphs = number_of_samples // number_of_models if equal_sizes else number_of_samples
    
    overall_graphs_generated = 0
    
    with Parallel(n_jobs=njobs) as workers:
            
        for model_label, configurations in models_and_configurations.items():
            for config in configurations:
                
                sub_label = get_label(model_label, config)
                            
                # try:
                generated_graphs = list(
                                        workers(
                                                delayed(MODEL_TO_FUNC[model_label])(n=number_of_nodes, **config) for _ in range(number_of_graphs)
                                                )
                                        )
                
                
                # convert graphs to networkx format
                if generated_graphs and not isinstance(generated_graphs[0], nx.classes.graph.Graph):
                    generated_graphs = list(
                                    workers(
                                                delayed(igraph_to_nx)(g) for g in generated_graphs
                                            )
                                            ) 


                
                precomputed_assets = precompute_assets_for_generated_graphs(generated_graphs=generated_graphs,
                                                                            distances_names_to_precompute_assets=distances_names_set,
                                                                            njobs=njobs,
                                                                            orca_path=orca_path)
                
                generated_graphs = [nx.to_numpy_array(g) for g in generated_graphs]
                
                
                if tmpdir is not None:
                    assert len(distances_names_set) == 1, "Currently support only single distance handling"
                    
                    dist = list(distances_names_set)[0]
                    
                    graph_filenames = []
                    precomputed_assets_filenames = []
                    
                    for i, graph in enumerate(generated_graphs):
                        
                        graph_adj_filename = os.path.join(tmpdir, f"{sub_label}_{i}")
                        entity_filename = os.path.join(tmpdir, f"_entity_{sub_label}_{i}")
                        
                        np.save(graph_adj_filename, graph)
                        np.save(entity_filename, precomputed_assets[dist][i])
                        
                        graph_filenames.append(graph_adj_filename + ".npy") # numpy autonaming.....
                        precomputed_assets_filenames.append(entity_filename + ".npy")
                        
                    
                    del generated_graphs
                    del precomputed_assets
                    generated_graphs = graph_filenames
                    precomputed_assets = {dist: precomputed_assets_filenames}
                
                overall_graphs_generated += len(generated_graphs)
                
                distances_names = ", ".join(sorted(distances_names_set))
                logger.info(f"Model {sub_label}. Generated graphs and precomputed assets for distances: {distances_names}")
                graphs[model_label][sub_label] = GraphModel(label=sub_label, graphs=generated_graphs, 
                                                            precomputed_descriptors=precomputed_assets,
                                                            )
                
        number_of_graphs_to_be_generated = number_of_samples - overall_graphs_generated
        
        if number_of_graphs_to_be_generated > 0:
            last_graphs = graphs[model_label][sub_label].graphs
            
            already_precomputed_amount = len(last_graphs)
            
            generated_graphs = list(
                                    workers(
                                            delayed(MODEL_TO_FUNC[model_label])(n=number_of_nodes, **config) for _ in range(number_of_graphs_to_be_generated)
                                            )
                                    )
                            
                            
            # convert graphs to networkx format
            if generated_graphs and not isinstance(generated_graphs[0], nx.classes.graph.Graph):
                generated_graphs = list(
                                workers(
                                            delayed(igraph_to_nx)(g) for g in generated_graphs
                                        )
                                        ) 


            
            precomputed_assets = precompute_assets_for_generated_graphs(generated_graphs=generated_graphs,
                                                                        distances_names_to_precompute_assets=distances_names_set,
                                                                        njobs=njobs,
                                                                        orca_path=orca_path)
            
            generated_graphs = [nx.to_numpy_array(g) for g in generated_graphs]
            
            
            if tmpdir is not None:
                assert len(distances_names_set) == 1, "Currently support only single distance handling"
                
                dist = list(distances_names_set)[0]
                
                graph_filenames = []
                precomputed_assets_filenames = []
                
                for i, graph in enumerate(generated_graphs, already_precomputed_amount):
                    
                    graph_adj_filename = os.path.join(tmpdir, f"{sub_label}_{i}")
                    entity_filename = os.path.join(tmpdir, f"_entity_{sub_label}_{i}")
                    
                    np.save(graph_adj_filename, graph)
                    np.save(entity_filename, precomputed_assets[dist][i])
                    
                    graph_filenames.append(graph_adj_filename + ".npy") # numpy autonaming.....
                    precomputed_assets_filenames.append(entity_filename + ".npy")
                    
                
                del generated_graphs
                del precomputed_assets
                generated_graphs = graph_filenames
                precomputed_assets = {dist: precomputed_assets_filenames}
            
            overall_graphs_generated += len(generated_graphs)
            
            
        graphs[model_label][sub_label].graphs.extend(generated_graphs)
        
        for dist in distances_names_set:
            graphs[model_label][sub_label].precomputed_descriptors[dist].extend(precomputed_assets[dist])
        
        assert overall_graphs_generated == number_of_samples
        
        
    return graphs

def get_initial_graphs(config, threads, distances_set, greedy_graphs_objects_per_distance, 
                       samples, nodes_number, orca_path, equal_sizes, 
                       maybe_ready_graphs, models_configurations=None, tmpdir=None,
                       ):
        # initial_graphs_preparation
        
    initial_graphs: Dict[str, 
                            Dict[Distance_name, List[Tuple[np.ndarray, str, Any]]]
                            ] = {}
    
    if "greedy" in config["initial_graphs"]:
        
        initial_graphs["greedy"] = greedy_graphs_objects_per_distance
    
    if "ER" in config["initial_graphs"]:
        er_graphs: GraphModel = list(list(generate_graphs(
                    models_and_configurations={"ER": [{"p": 0.5}]},
                    number_of_samples=samples,
                    number_of_nodes=nodes_number,
                    njobs=threads,
                    orca_path=orca_path,
                    distances_names_set=distances_set,
                    equal_sizes=equal_sizes,
                    tmpdir=tmpdir
                        ).values())[0].values())[0]
        
        # breakpoint()
        
        labels = list(map(lambda x: f"ER_{x}", range(samples)))
        graphs = er_graphs.graphs
        descriptors = er_graphs.precomputed_descriptors
        
        
        initial_graphs_er = {}
        for distance_name, assets in descriptors.items():
            
            g_l_e = zip(graphs.copy(), labels.copy(), assets.copy())
            
            initial_graphs_er[distance_name] = [(g, l, e) for g, l, e in g_l_e]
        
        initial_graphs["ER"] = initial_graphs_er
    
    
    if "mix" in config["initial_graphs"]:

        mix_graphs: Graphs_dict = generate_graphs(
                    models_and_configurations=config["models"],
                    number_of_samples=config["greedy_sampling_size"],
                    number_of_nodes=nodes_number,
                    njobs=threads,
                    orca_path=orca_path,
                    distances_names_set=distances_set,
                    equal_sizes=equal_sizes,
                    tmpdir=tmpdir,
                        )        
        initial_graphs_mix = defaultdict(list)

        for graph_model, submodels_dict in mix_graphs.items():
            
            for submodel_name, graphmodel in submodels_dict.items():
                graphs_graphmodel = graphmodel.graphs
                labels_graphmodel = list(map(lambda x: f"{submodel_name}_{x}", range(len(graphs_graphmodel))))
                descriptors_graphmodel = graphmodel.precomputed_descriptors
                
                for distance_name, assets in descriptors_graphmodel.items():
        
                    g_l_e = zip(graphs_graphmodel.copy(), 
                                labels_graphmodel.copy(), 
                                assets.copy(),
                                )
                    
                    initial_graphs_mix[distance_name].extend([(g, l, e) for g, l, e in g_l_e])
            
        initial_graphs["mix"] = initial_graphs_mix

    if "none" in config["initial_graphs"]:
        initial_graphs["none"] = dict((distance, []) for distance in distances_set)
        
    if "user" in config["initial_graphs"]:
        if maybe_ready_graphs is None:
            raise ArgumentError("User-defined initial mode provided, but graphs were not specified")
        
        user_graphmodel: GraphModel = transform_generated_graphs_dict_to_list(generated_graphs=maybe_ready_graphs,
                                                                   distances_set=distances_set,
                                                                   orca_path=orca_path,
                                                                   njobs=threads)[0]
        initial_graphs_user = {}
        
        graphs = user_graphmodel.graphs
        descriptors = user_graphmodel.precomputed_descriptors
        labels = list(map(lambda x: f"user_{x}", range(len(graphs))))
        for distance_name, assets in descriptors.items():
            
            g_l_e = zip(graphs.copy(), labels.copy(), assets.copy())
            
            initial_graphs_user[distance_name] = [(g, l, e) for g, l, e in g_l_e]
        
        initial_graphs["user"] = initial_graphs_user

    return initial_graphs
