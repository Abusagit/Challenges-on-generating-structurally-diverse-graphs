import numpy as np
import heapq
import networkx as nx
import logging
import pandas as pd

from queue import PriorityQueue
from joblib import Parallel, delayed
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
from itertools import combinations, product
from functools import partial, total_ordering
from tqdm import tqdm, trange
from collections import defaultdict

from base import DiversityBaseClass, GraphObject
from utils import get_label, time_wrapper, transform_generated_graphs_dict_to_list, GraphModel, Graph, save_pickle
from utils import Graph_representation, Distance_func, Graph_random_model, Distance_name, Parameters, Metainfo, Report
from generation import generate_graphs
from distances import DISTANCE_TO_FUNC, precompute_assets_for_generated_graphs


# cache for computed distances between graphs each graph distance:


logger = logging.getLogger(__name__)

@total_ordering
class PQEntry: # TODO refactor
    """
    Interface for entry in Priority queue for graph greedy max volume generation
    
    """
    __slots__ = ("index", "graph_object", "cumulative_distance")
    def __init__(self, 
                 graph_object: GraphObject, 
                 cumulative_distance=0) -> None:
        
        self.graph_object: GraphObject = graph_object
        self.cumulative_distance: float = cumulative_distance
    
    def __eq__(self, entry) -> bool:
        return self.cumulative_distance == entry.cumulative_dustance
    
    def __lt__(self, entry) -> bool: # reverse - because we use implemented in python min PQ, but need max PQ
        return self.cumulative_distance > entry.cumulative_distance



class GreedyOptimizer(DiversityBaseClass):
    # NOTE GraphObjects are sorted in increasing order of theior fitnesses
    def __init__(self, 
                 initial_population: List[Tuple[np.ndarray, str, Any]], 
                 distance_name: Distance_name, 
                 distance_function: Distance_func,
                 volume_func_type: str, 
                 number_of_graphs:int,
                 workers_options: Dict[str, Any],
                 *args, **kwargs) -> None:
        DEFAULT_CUMULATIVE_SUM = {
            "avg": 0.0,
            "min": float("inf"),
            "energy": 0.0,
        }
                
        
        super().__init__(initial_population=initial_population, 
                         distance_name=distance_name, 
                         distance_function=distance_function,
                         volume_func_type=volume_func_type, 
                         number_of_graphs=number_of_graphs,
                         MODE="greedy", 
                         *args, 
                         **kwargs
                         )
        
        class MaxPQAvg(PriorityQueue):
            """
            class for performing greedy algorithm of generating maximally diverse dataset
            """
            
            def __init__(self, queue: List[PQEntry], parallel_workers: Parallel, greedy_optimizer: GreedyOptimizer, maxsize=0) -> None:
                super().__init__(maxsize)
                
                # heapify inplace - O(n)
                self.greedy_optimizer: GreedyOptimizer = greedy_optimizer
                self.queue: List[PQEntry] = queue
                self.parallel_workers: Parallel = parallel_workers
                heapq.heapify(self.queue)
                
                
            def update_queue(self, new_entry_in_max_vol_set: PQEntry):
                """
                updates entries in self.queue according to their distance to <new_entry_in_max_vol_set>
                
                distance_metric: function for computing distance
                new_entry_in_max_vol_set: greedily chosen entry wich maximizes objective
                distance_name: label for distance for access to cache
                """
                    
                self.parallel_workers(
                    delayed(self._update)(i, new_entry_in_max_vol_set.graph_object) for i in range(len(self.queue))
                )
                
                # O(n)
                heapq.heapify(self.queue)
            
            def _update(self, 
                        index: int, 
                        graph_object: GraphObject
                        ):
                """
                update entries in priority queue - function for parallel computing
                
                index: index of element to be updated
                distance_metric: function for computing distance
                label_representation_2: representation of max_vol_entry for current distance_name
                distance_name: access to cache
                """
                
                self.queue[index].cumulative_distance += self.greedy_optimizer._graph_distance(
                    self.queue[index].graph_object,
                    graph_object,
                    save_in_cache=False,
                )
                
            def top(self) -> PQEntry:
                """
                returns greedily the most satisfying element
                """
                return heapq.heappop(self.queue)
            
        class MaxPQMin(MaxPQAvg):
            def __init__(self, **kwargs) -> None:
                super().__init__(**kwargs)
                
            def _update(self, 
                        index: int, 
                        graph_object: GraphObject
                        ):
                """
                Updates queue values priorities according to minimum property
                """
                
                self.queue[index].cumulative_distance = min(self.greedy_optimizer._graph_distance(
                                                                    self.queue[index].graph_object,
                                                                    graph_object,
                                                                    save_in_cache=False,
                                                                ),
                                                            self.queue[index].cumulative_distance,
                                                        )
                
        class MaxPQEnergy(MaxPQAvg):
            def __init__(self, **kwargs) -> None:
                super().__init__(**kwargs)
                
            def _update(self, 
                        index: int, 
                        graph_object: GraphObject
                        ):
                """
                Updates queue values priorities according to minimum property
                """
                
                self.queue[index].cumulative_distance -=  self.greedy_optimizer._graph_distance(
                                                                        self.queue[index].graph_object,
                                                                        graph_object,
                                                                        save_in_cache=False,
                                                                        )

        if volume_func_type == "avg":
            self.GraphsPQ = MaxPQAvg
        elif volume_func_type == "min":
            self.GraphsPQ = MaxPQMin
        elif volume_func_type == "energy":
            self.GraphsPQ = MaxPQEnergy
        else:
            raise NotImplementedError

        self.initial_population = [
            PQEntry(
                graph_object=G,
                cumulative_distance=DEFAULT_CUMULATIVE_SUM[volume_func_type]
            ) for G in self.population
        ]
        self.workers_options = workers_options

        
    def greedy_algorithm(self) -> Tuple[List[np.ndarray], Tuple[np.ndarray, str, Any], float]:
        with Parallel(**self.workers_options) as parallel_workers:
            max_pq_graphs = self.GraphsPQ(queue=self.initial_population,
                                        parallel_workers=parallel_workers,
                                        greedy_optimizer=self,
                                        )
            
            max_volume_set: List[PQEntry] = []
            
            for _ in trange(self.number_of_graphs,
                          total=self.number_of_graphs,
                          desc=f"Generating max diverse set for {self.distance_name}"
                          ):
                
                max_vol_entry: PQEntry = max_pq_graphs.top()
                max_pq_graphs.update_queue(
                    new_entry_in_max_vol_set=max_vol_entry
                )
                
                max_volume_set.append(max_vol_entry)
                
        
            greedy_graph_objects: List[GraphObject] = [entry.graph_object for entry in max_volume_set]
            self.population = greedy_graph_objects
            
            fitness = self.recompute_and_return_fitness(workers=parallel_workers)
        
        return self.get_graphs(), [(G.graph, G.identification, G.entity) for G in self.population], fitness

def run_configuration(initial_population: List[GraphModel],
                      number_of_graphs:int,
                      distance_name: str,
                      volume_func_type:str,
                      initial_graphs_label: str,
                      save_dir:Path,
                      run_number: int,
                      workers_options: Dict[str, Any],

                      ):
    checkpoint_dir = save_dir / "checkpoints" / str(run_number)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    distance_function = DISTANCE_TO_FUNC[distance_name]

    optimizer = GreedyOptimizer(initial_population=initial_population,
                    number_of_graphs=number_of_graphs,
                    distance_function=distance_function,
                    distance_name=distance_name,
                    volume_func_type=volume_func_type,
                    checkpoint_dir=checkpoint_dir,
                    workers_options=workers_options
                    )
    
    greedy_graphs, greedy_graphs_objects, fitness_finish = optimizer.greedy_algorithm()
    
    save_pickle(
        obj=greedy_graphs,
        to_filename=checkpoint_dir/"greedy_graphs.pkl",
    )
    
    data = [run_number, distance_name, initial_graphs_label, fitness_finish]
    del optimizer
    return data, greedy_graphs_objects
    
    
    
def generate_max_volume_greedy(
                                initial_graphs: Dict[str, List[Tuple[np.ndarray, str, Any]]],
                                number_of_graphs: int,
                                distances_configs: Dict[Distance_name, List[Parameters]],
                                n_jobs: int,
                                volume_func_type: str,
                                save_dir: Path or None,
                                ) -> Dict[str, List[Report]]:

    # TODO rewrite description
    """
    Greedy algorithm for generating maximally diverse dataset
    
    
    models_configs_for_greedy: dict of parameters for initial models pool, from which maximally diverse set will be generated
    number_of_graphs: number of resulting graphs in output
    number_of_vertices: number of nodes in [every] graph (so far every graph has equal amount of nodes)
    distances_configs: lists of configurations for each distance
    n_jobs: number of threads
    k_fold_larger: ratio
    init_types: types of initialization for each run. Can be list of parameters or single paramtetrs
        "ER" - first graph in diverse set will be Erdos-Renyi model with p=0.5
        "smart" - first graph in diverse set will be chosen from generated pool of graphs by the following: 
                    * let N be size of overall set of generated graphs
                    * sample randomly k_i * N graphs from this set (0 < k <= 1)
                    * choose graph which hase largest distance to any graph in this subset
                    
    smart_init_ratios: fraction(s) k_i used for "smart" initializations
    equal_sizes: whether or not generate full size of graphs for each graph random model. 
            If True, for each random model configuration will be generated `number_of_graphs` // M graphs, where M is the number of configurations for given random model
            If False, for each random model configuration will be generated exactly `number_of_graphs` graphs
    graphs: precomputed graphs or empty set, if the latter is true, graphs will be generated here

    
    returns list of report for each init_type (can be multiple runs for some type)
     
    """
    
    # TODO need to be a list!! in the future
    
    workers_options = dict(
        n_jobs=n_jobs,
        batch_size="auto", 
        backend="threading",
    )
    
    DATA = []
    COLUMNS = ["Run idx", "Distance", "Initial population", "Volume"]
        
    parameters_grid = product(distances_configs.keys(), initial_graphs.items())
    
    # save graphs set with best fitness for each distance
    
    greedy_graphs_per_distance: Dict[str, Tuple[np.ndarray, str, Any]] = {}
        
    for run_number, [distance_name, (initial_graphs_label, initial_graphs_dict)] in enumerate(parameters_grid):
        report_list, graphs_objects = run_configuration(
                        initial_population=initial_graphs_dict[distance_name],
                        number_of_graphs=number_of_graphs,
                        distance_name=distance_name,
                        volume_func_type=volume_func_type,
                        initial_graphs_label=initial_graphs_label,
                        run_number=run_number,
                        save_dir=save_dir,
                        workers_options=workers_options,
                    )
        
        greedy_graphs_per_distance[distance_name] = graphs_objects

        DATA.append(report_list)
            
    
    report_dataframe = pd.DataFrame(data=DATA, columns=COLUMNS)
    
    report_dataframe.to_csv(save_dir / "greedy_report.csv")
    
    return report_dataframe, greedy_graphs_per_distance