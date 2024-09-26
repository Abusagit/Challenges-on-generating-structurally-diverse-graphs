import logging
import numpy as np
import networkx as nx
import pandas as pd

from pathlib import Path
from typing import Dict, List, Any, Tuple, Callable, MutableSet, Iterable, Union
from joblib import Parallel, delayed
from collections import defaultdict
from math import floor, ceil, sqrt
from scipy.stats import bernoulli
from tqdm import trange
from itertools import product
from functools import partial

from distances import PRECOMPUTE_DICT, ProgressParallel
from greedy import DISTANCE_TO_FUNC
from base import GraphObject, DiversityBaseClass
from utils import Distance_func, Distance_name, Graph, Parameters, get_label, save_pickle, time_wrapper

logger: logging.Logger = logging.getLogger(__name__)

ProbabilityFunc = Callable[[float, float], float]
TemperatureFunc = Callable[[int], float]
GraphDistanceFunc = Callable[[GraphObject, GraphObject], float]

def save_checkpoint(save_subdir, graphs, fitnesses, all_fitnesses, volumes):
    save_pickle(obj=graphs, to_filename=save_subdir / "graphs.pkl")
    save_pickle(obj=fitnesses, to_filename=save_subdir / "fitnesses.pkl")
    save_pickle(obj=all_fitnesses, to_filename=save_subdir / "all_fitnesses.pkl")
    save_pickle(obj=volumes, to_filename=save_subdir / "volumes.pkl")


class LocalOptimizer(DiversityBaseClass):
    
    def __init__(self,
                 n_jobs:int,
                 initial_graphs: str,


                 
                 distance_name: Distance_name,
                 distance_function: Distance_func,
                 number_of_nodes: int,
                 number_of_graphs: int,
                 total_attempts: int,
                 initial_population: List[GraphObject],
                 checkpoint_dir:Path,
                 volume_func_type:str,
                 max_failed_attempts: int=100,
                 checkpoint_interval:int=100,

                 parallel_backend:str="threading",
                 directed: bool = False,

                 ) -> None:
        
        super().__init__(
            initial_population=initial_population,
            distance_name=distance_name,
            distance_function=distance_function,
            number_of_nodes=number_of_nodes,
            number_of_graphs=number_of_graphs,
            total_attempts=total_attempts,
            checkpoint_dir=checkpoint_dir,
            volume_func_type=volume_func_type,
            max_failed_attempts=max_failed_attempts,
            checkpoint_interval=checkpoint_interval,
            MODE="local"
        )

        self.precompute_entity_func: Callable[..., Any] = PRECOMPUTE_DICT[distance_name]
        
        
        self.n_jobs: int = n_jobs
        self.parallel_backend: str = parallel_backend
                
        # self.us
        self.N_SQUARED = (self.number_of_nodes - 1) * (self.number_of_nodes - 2) // 2
        
        
        self.initial_graphs_mode: str = initial_graphs
        
        self.successful_fitnesses = []
        self.all_fitnesses = []
        
        self.directed = directed
    
    def generate_random_graph_object(self, identification: str):
        
        
        graph = nx.random_graphs.gnp_random_graph(n=self.number_of_nodes, 
                                                  p=0.5, 
                                                #   seed=self.seed,
                                                  )
    
        graph_object = self.get_graph_object_from_graph_and_id(graph, identification=identification)
        
        return graph_object
    
    def get_graph_neighbor(self, graphobj: GraphObject, identification: str):
                
        i, j = np.random.choice(self.number_of_nodes, size=2, replace=False)
        i, j = min(i, j), max(i, j)        

        adj_matrix = graphobj.graph.copy()
        
        # flip the value at A_ij
        adj_matrix[i, j] = not adj_matrix[i, j]
        
        if not self.directed:
            adj_matrix[j, i] = not adj_matrix[j, i]
        
        
        new_graph_object = self.get_graph_object_from_graph_and_id(graph=adj_matrix, identification=identification)
        
        return new_graph_object            
            
            
    def distances_from_chosen_graph(self, current_candidate: GraphObject, parallel_workers: Parallel, index: int):
        
        distances_maybe_before = parallel_workers(
                delayed(self._graph_distance)(current_candidate, graph) for graph in self.population[:index]
            )
        
        distances_maybe_after = parallel_workers(
                delayed(self._graph_distance)(current_candidate, graph) for graph in self.population[index+1:]
            )
        
        return distances_maybe_before, distances_maybe_after
        
    
    def get_first_graph_and_its_index(self, parallel_workers: Parallel, graph_number:int) -> Tuple[GraphObject, float]:
        
        if self.initial_graphs_mode == "none":
            graph_best = self.generate_random_graph_object(f"{graph_number}_0")
            f_best = self.graph_object_default_fitness_value
        else:
            graph_best = self.population[graph_number]
            
            # TODO sample from graphs
            f_best = graph_best.fitness

        
        return graph_best, f_best
        

    
    def calculate_fitness_of_new_candidate(self, graph_candidate: GraphObject, parallel_workers: Parallel, index_to_exclude:int,
                                            ):
        
        distances_to_the_left_of_index, distances_to_the_right_of_index = self.distances_from_chosen_graph(current_candidate=graph_candidate, 
                                                      parallel_workers=parallel_workers,
                                                      index=index_to_exclude,
                                                      )
        if self.volume_func_type == "energy" or self.volume_func_type == "avg":
            left_sum = 0 if len(distances_to_the_left_of_index) == 0 else np.sum(distances_to_the_left_of_index)
            right_sum = 0 if len(distances_to_the_right_of_index) == 0 else np.sum(distances_to_the_right_of_index)
            overall_sum_distances = left_sum + right_sum
            
            return overall_sum_distances
                    
        if self.volume_func_type == "min":
            if len(distances_to_the_left_of_index) == 0:
                return min(distances_to_the_right_of_index)
            if len(distances_to_the_right_of_index) == 0:
                return min(distances_to_the_left_of_index)
            
            return min(
                min(distances_to_the_right_of_index),
                min(distances_to_the_left_of_index)
            )
        
    
    @time_wrapper
    def annealing(self):
        description_string = f"Performing local combinatorial optimization of {self.number_of_graphs} graphs woth {self.distance_name} distance with {self.total_attempts} attempts"
        
        with Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend) as parallel_workers:
            
            volume = self.recompute_and_return_fitness(parallel_workers)
            self.volumes.append(volume)
            
            fitnesses, probabilities = self.get_list_of_fitnesses_and_probabilities()
            
            fitnesses_sum = sum(fitnesses)

            is_attempt_successful = False
            
            attempts_without_growth = 0
            
            
            diff_best = None
            f_best = None
            index_to_delete_for_best = None
            best_graph_for_change = None
            
            for attempt in trange(1, self.total_attempts + 1, 
                                  desc=description_string):
                

                
                index_of_weak = np.random.choice(self.number_of_graphs, 1, p=probabilities)[0]
                changed_graph = self.get_graph_neighbor(graphobj=self.population[index_of_weak], identification=f"attempt_{attempt}")
                
                f_new = self.calculate_fitness_of_new_candidate(changed_graph, parallel_workers, index_of_weak)
                diff_now = f_new - self.population[index_of_weak].fitness
                                
                if self.volume_func_type == "energy":
                    diff_now *= -1

                
                if diff_best is None:
                    diff_best = diff_now
                    f_best = f_new
                    index_to_delete_for_best = index_of_weak
                    best_graph_for_change = changed_graph
                
                if diff_now > 0:
                    is_attempt_successful = True
                else:
                    is_attempt_successful = False
                    attempts_without_growth += 1
                    
                    if diff_now > diff_best:
                        diff_best = diff_now
                        f_best = f_new
                        index_to_delete_for_best = index_of_weak
                        best_graph_for_change = changed_graph 



                if is_attempt_successful:
                    self.population[index_of_weak] = changed_graph
                    self.population[index_of_weak].fitness = f_new

                    volume = self.recompute_and_return_fitness(parallel_workers)
                    fitnesses, probabilities = self.get_list_of_fitnesses_and_probabilities()

                    self.successful_fitnesses.append(1)
                    
                    
                    diff_best = None
                    f_best = None
                    index_to_delete_for_best = None
                    best_graph_for_change = None
                    attempts_without_growth = 0
                    
                elif attempts_without_growth == self.max_failed_attempts:

                    self.population[index_to_delete_for_best] = best_graph_for_change
                    self.population[index_to_delete_for_best].fitness = f_best

                    volume = self.recompute_and_return_fitness(parallel_workers)
                    fitnesses, probabilities = self.get_list_of_fitnesses_and_probabilities()

                    self.successful_fitnesses.append(1)
                    
                    
                    diff_best = None
                    f_best = None
                    index_to_delete_for_best = None
                    best_graph_for_change = None
                    attempts_without_growth = 0
                    
                    
                else:
                    self.successful_fitnesses.append(0)
                
                self.volumes.append(volume)

                if attempt % 1000 == 0:
                    # serialize intermediate result        
                    save_checkpoint(self.checkpoint_dir, self.get_graphs(), self.successful_fitnesses, self.all_fitnesses, self.volumes)
                    
                if attempt % 10000 == 0:
                    self._purge_cache()
 
        save_checkpoint(self.checkpoint_dir, self.get_graphs(), self.successful_fitnesses, self.all_fitnesses, self.volumes)

        return volume, max(self.volumes)
        


def run_configuration(save_dir: Path,
                      run_number: int, 
                      volume_func_type:str,
                      distance_name_subname_function: Tuple[str, str, Callable[[Graph, Graph], float]],
                      parameter_options: Tuple[str, Any],
                      common_arguments: Dict[str, Any],
                      initial_graphs: Dict[str, Dict[Distance_name, List[GraphObject]]],
                      ):
    
    distance_name, distance_label, distance_function = distance_name_subname_function  
      
    
    save_subdir = save_dir / f"{run_number}"
    
    
    parameters = dict(parameter_options)
    parameters.update(common_arguments)
    parameters.update(dict(
        distance_name=distance_name,
        distance_function=distance_function,
    ))
    
    
    initial_population = initial_graphs[parameters["initial_graphs"]][distance_name].copy()
    
    logger.info(f"Initializing optimizer for run {run_number}, distance {distance_label}, arguments {parameters}")
    save_subdir.mkdir(parents=True, exist_ok=True)

    optimizer = LocalOptimizer(**parameters, 
                                       initial_population=initial_population,
                                       checkpoint_dir=save_subdir,
                                       volume_func_type=volume_func_type,
                                       )
    
    
    volume, max_volume = optimizer.annealing()
    
    data_from_run = [run_number, distance_label] + [param for _, param in parameter_options] + [volume, max_volume]
    
    logger.info(
        f"Ran {parameters}, volume {volume}"
    )
    
    
    return data_from_run

def optimize_set_of_graphs_locally(config: Dict[str, Any],
                                         volume_func_type:str,
                                         number_of_nodes: int,
                                         number_of_graphs: int, 
                                         distances_configurations: Dict[Distance_name, List[Parameters]],
                                         n_jobs: int,
                                         save_dir: Path or None,
                                         orca_path: Path,
                                         initial_graphs: Dict[str, Dict[Distance_name, List[GraphObject]]]
                                         ) -> Dict[str, Dict[str, List[Graph]]]:
    
    
    PRECOMPUTE_DICT["GCD"] = partial(PRECOMPUTE_DICT["GCD"], orca_path=orca_path)
    logger.info("Initialized main function")
    
    distances_names_subnames_functions = []
    
    for distance_name, params_list in distances_configurations.items():
        for params in params_list:
            distances_names_subnames_functions.append([distance_name, 
                                                       get_label(distance_name, params), 
                                                       DISTANCE_TO_FUNC[distance_name]
                                                    ]
                                                )
            
    logger.info("Initialized distances functions")
    
    parameters_with_names = []
    for parameter_name, parameters_list in config.items():
        parameters_with_names.append(
            [
                (parameter_name, param) for param in parameters_list
            ]
        )
        
    parameters_product  = tuple(
        product(
            *parameters_with_names
        )
    )
    logger.info("Initialized parameters combinations")
    
    distances_and_parameters_combinations = enumerate(
                                                    product(
                                                        distances_names_subnames_functions, parameters_product
                                                    ), 0)
        
    logger.info("Initialized combinations of distances and parameters")
    
    COLS_FOR_REPORT_TABLE = ["run", "distance"] + [params[0][0] for params in parameters_with_names] + ["volume_finish", "volume_max"]
    common_args = dict(
        number_of_nodes=number_of_nodes,
        number_of_graphs=number_of_graphs,
        n_jobs=n_jobs,
    )
    
    with ProgressParallel(n_jobs=n_jobs, backend="multiprocessing") as parallel_processors:
        
        runs_data = parallel_processors(
            delayed(run_configuration)(save_dir, 
                                       run_number,
                                       volume_func_type,
                                       distance_name_subname_func,
                                       parameter_options,
                                       common_args,
                                       initial_graphs,
                                       ) for run_number, (distance_name_subname_func, parameter_options) in distances_and_parameters_combinations
        )
        
    
    logger.info("Finished every run")
    
    report_df = pd.DataFrame(
        data=runs_data,
        columns=COLS_FOR_REPORT_TABLE,
    )
    
    return report_df