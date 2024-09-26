__doc__ = """
Implementation of genetic algorithm.

Note 1: here in the code we use term 'volume' as a measure of how diverse given set of graphs, and so far it can be measured as: 
        1) average pairwise distance (called 'diversity' in some of papers and thus can lead to misunderstandins) - this options is called 'avg'
        2) minimal pairwise distance (called 'bottleneck' in some papers) - this options is called 'min'
Note 2: '# type: ignore' comments disable type checking in current line, because some methods are reassigned through hashtable and Pylance can`t resolve it.
"""

from base import GraphObject, DiversityBaseClass
from utils import time_wrapper
from utils import Distance_func, Distance_name, Graph, Parameters, get_label, save_pickle

from typing import Dict, List, Any, Tuple, Callable, Union, MutableSet, Iterable

import pandas as pd

import logging

from joblib import Parallel, delayed
from scipy.stats import bernoulli
import networkx as nx
from functools import partial, reduce
import numpy as np

import random
from itertools import combinations, product
from collections import defaultdict
from distances import PRECOMPUTE_DICT
from greedy import DISTANCE_TO_FUNC

from pathlib import Path


from tqdm import tqdm


logger: logging.Logger = logging.getLogger(__name__)



class GraphsPopulation(DiversityBaseClass):
    
    def __init__(self, 
                workers_options: Dict[str, Any],
                mutation_prob:float,                 
                
                 initial_graphs:List[GraphObject],
                 total_attempts: int,
                 distance_name: Distance_name,
                 distance_function: Distance_func,
                 max_failed_attempts:int,                  
                 checkpoint_interval:int,
                 volume_func_type: str,
                 checkpoint_dir:Path,
                 **kwargs,
                 
                 ) -> None:
        
        super().__init__(
            initial_population=initial_graphs,
            distance_name=distance_name,
            distance_function=distance_function,
            total_attempts=total_attempts,
            max_failed_attempts=max_failed_attempts,
            checkpoint_interval=checkpoint_interval,
            volume_func_type=volume_func_type,
            checkpoint_dir=checkpoint_dir,
            MODE="genetic",
            **kwargs,
        )
        
        
        self.crossover_func = partial(self._crossover_assignment, assignment_func=self._random_assignment)
        
        # assign functions depending on parameters
        self.mutation_func = self._vertex_mutation
        self.mutation_prob = bernoulli(mutation_prob)
        
        
        # self.parallel_workers = parallel_workers
        self.workers_options = workers_options
        self.max_failed_attempts = max_failed_attempts

        self.graphs_generated: List[int] = []
        self.children_checked = 0
                
        

        self.triu_mask = np.ones((self.number_of_nodes, self.number_of_nodes))
        self.triu_mask[np.tril_indices_from(self.triu_mask)] = 0
        

    @property
    def metainfo(self):
        return dict(
            population=self.get_graphs_from_population(self.population),
            volumes=self.volumes,
            graphs_generated=self.graphs_generated,
        )

    def get_parents_proportionally_to_fitness(self, population_fitnesses: List[float]) -> List[Tuple[GraphObject, GraphObject]]:
        """
        Takes list of non-negative numbers, where i-th element of the list
        is the fitness value of i-th graph. 
        Returns an array with two different indices
        """
        # Normalize the list
        lst_norm = np.array(population_fitnesses) / np.sum(population_fitnesses)  

        # Sample 2 indices from the distribution without replacement
        
        parents = (np.random.choice(self.population, size=2, replace=False, p=lst_norm) for _ in range(self.max_failed_attempts))
        
        return parents
        
        
    @staticmethod
    def _random_assignment(parent_1: np.ndarray, parent_2: np.ndarray, **kwargs) -> List[int]:
        """
        For each pairs of graphs takes two graphs size, returns an array of 0s and 1s.
        We call this array an "assignment".
        Assignment says which vertices we take from which parents.
        Assignment is used in crossover.
        """
        
        number_of_nodes = parent_1.shape[0]

        return list(np.random.choice([1, 2], size=number_of_nodes))
    
    def _crossover_assignment(self, parent_1: np.ndarray, parent_2: np.ndarray, 
                              assignment_func: Callable[[np.ndarray, np.ndarray], 
                                                        List[int]
                                                        ]) -> np.ndarray:
        
        
        assignment = assignment_func(parent_1=parent_1, parent_2=parent_2) # type: ignore
        """
        Takes two networkx graphs and an array of 1s and 2s.
        Returns the networkx graph - descendant.
        """

        #create descendant with right number of nodes and no edges
        descendant = np.zeros_like(parent_1)

        """
        For each pair (i,j) we decide, do we add edge (i,j) to the descendant or not. 
        If in the assignment both vertices i and j are taken from the same parent 
        then we copy (i,j) edge/noedge of this parent.
        If i and j are from the different parents - we randomly choose one of the parents
        (with p = 0.5) and copy (i,j) edge/noedge of this parent. 
        """
        
        edges = set(zip(*(parent_1 * self.triu_mask).nonzero())) | set(zip(*(parent_2 * self.triu_mask).nonzero()))

        for i, j in edges:
            if assignment[i] == 1 and assignment[j] == 1:
                if parent_1[i, j]: 
                    descendant[i, j] = descendant[j, i] = 1

            elif assignment[i] == 2 and assignment[j] == 2:
                if parent_2[i, j]: 
                    descendant[i, j]= descendant[j, i] = 1

            else:
                if random.randint(1, 2) == 1:
                    if parent_1[i, j]: 
                        descendant[i, j] = descendant[j, i] = 1
                else:
                    if parent_2[i, j]: 
                        descendant[i, j] = descendant[j, i] = 1

        return descendant
    
    def _vertex_mutation(self, G:np.ndarray) -> np.ndarray: 
        """
        Takes networkx graph. Chooses random vertex, deletes all edges from it,
        adds random number of random edges from this vertex. 
        Returns the networkx graph.
        """
        # Choose a random vertex from the graph and remove its edges
                
        
        vertex = np.random.choice(self.number_of_nodes)
        A = G.copy()
        
        A[vertex, :] = 0
        A[:, vertex] = 0
        
        # Choose a random number of edges to add 
        num_edges = random.randint(1, self.number_of_nodes - 1)

        # Choose random vertices to add edges to
        other_vertices = set(range(self.number_of_nodes)) - {vertex}

        random_vertices = np.random.choice(list(other_vertices), num_edges, replace=False)

        # Add edges from the deleted vertex to the random vertices
        
        # NOTE can crash (?)
        A[vertex, random_vertices] = 1
        A[random_vertices, vertex] = 1

        # Return the modified graph
        return A

    def crossover_mini(self, i: int, parent_1: GraphObject, parent_2: GraphObject, iteration: int):
        # print(f"Step {iteration}, parent_1: {parent_1.identification}, parent_2: {parent_2.identification}")
        child_id_pattern = "child_num_{}_iteration_{}"
        
        child_name: str = child_id_pattern.format(i, iteration)
        child_graph: np.ndarray = self.crossover_func(parent_1.graph, parent_2.graph) # type: ignore
        
        return child_name, child_graph
    
    def maybe_mutate(self, G: Tuple[str, np.ndarray], indicator: int):
        graph_name, graph = G
        
        graph = self.mutation_func(G=graph) if indicator else graph
        
        graph_object = self.get_graph_object_from_graph_and_id(graph=graph, identification=graph_name)
        
        return graph_object

    def _preprocess_before_attempt(self, index_of_weak: int):
        if self.volume_func_type == "avg" or self.volume_func_type == "energy":
            min_distance_without_weak = 0
            
            weak_graph_vol_norm = 2 * self.population[index_of_weak].fitness

        elif self.volume_func_type == "min":
            weak_graph_vol_norm = 0
            
            min_distance_without_weak = float("inf")
            # indices = set(range(self.N))
            
            for j, graph in enumerate(self.population):
                graph._prev_fitness = graph.fitness
                
                if j == index_of_weak:
                    continue
                
                if graph.distances_to_population is None:
                    graph.distances_to_population = [self._graph_distance(graph, g) for g in self.population]
                    graph.distances_to_population[j] = self.graph_object_default_fitness_value
                
                
                cursed_distance, graph.distances_to_population[index_of_weak] = graph.distances_to_population[index_of_weak], self.graph_object_default_fitness_value
                
                if cursed_distance <= graph.fitness:
                    graph.fitness  = min(graph.distances_to_population)
                graph.distances_to_population[index_of_weak] = cursed_distance
                
                min_distance_without_weak = min(min_distance_without_weak, graph.fitness)
                
            
            
        return min_distance_without_weak, weak_graph_vol_norm
    def evolution_step_wrapper(self, population_fitnesses: List[float], 
                               step_number: int, 
                               current_volume: float, 
                               ) -> Tuple[float, float]:
        """
        Evolution step functional. Updates population, tries to mutate successive child and stores metainformation
        
        """
        
        evolution_step_results: Dict[str, Any] = self.evolution_step(population_fitnesses=population_fitnesses, 
                                                                     step_number=step_number,
                                                                     current_volume=current_volume,
                                                                    )
        
        successive_child, new_volume = evolution_step_results["max_child"], evolution_step_results["max_vol"]
        
        
        max_weak_index = evolution_step_results["max_weak_index"]
        self.update_population(weak_graph_index=max_weak_index, new_graph=successive_child)
        
        return new_volume
    
    def _preprocess_before_evolution_step(self, population_fitnesses: List[float]):
        
        if self.volume_func_type == "avg" or self.volume_func_type == "energy":
            total_cumulative_sum_remaining = reduce(lambda x, y: x + y.fitness, self.population, 0)
            indices_to_try_as_weak = list(range(self.N))
            
            # NOTE this is temporary
            # indices_to_try_as_weak = (self.eradication_func(population_fitnesses=population_fitnesses) for _ in range(2))#self.population)
            
        elif self.volume_func_type == "min":
            total_cumulative_sum_remaining = 0
            indices_to_try_as_weak = list(self.guys_achieving_minimal_distance)
            
            if (x := len(indices_to_try_as_weak)) > 2:
                logger.warning(f"Found {x} graphs contributing to the minimal distance!")
        else:
            raise ValueError(f"Volume {self.volume_func_type} is incorrect name")
            
            
        return total_cumulative_sum_remaining, np.random.permutation(indices_to_try_as_weak)
        
    def _postprocess_after_evolution_step(self, total_cumulative_sum_remaining, weak_graph_vol_norm, max_vol):
        
        if self.volume_func_type == "avg" or self.volume_func_type == "energy":
            current_cumsum = total_cumulative_sum_remaining - weak_graph_vol_norm
            
        elif self.volume_func_type == "min":
            current_cumsum = max_vol
        else:
            raise ValueError(f"{self.volume_func_type}")
            
        return current_cumsum
    
    def _postprocess_after_attempt(self, index_of_weak: int, weak_graph: GraphObject):
        if self.volume_func_type == "min":
            
            for graph in self.population:
                graph.fitness = graph._prev_fitness
                
    def generate_children(self,step_number:int, population_fitnesses: List[float]):
        
        for i, (parent_1, parent_2) in enumerate(self.get_parents_proportionally_to_fitness(population_fitnesses=population_fitnesses)):
            mutation_indicator = self.mutation_prob.rvs()
            child_name_graph = self.crossover_mini(
                    i=i,
                    parent_1=parent_1,
                    parent_2=parent_2,
                    iteration=step_number,
                )
            
            child = self.maybe_mutate(child_name_graph, mutation_indicator)
            
            yield child
      
    def evolution_step(self,
                        population_fitnesses: List[float], 
                        step_number: int, 
                        current_volume: float, 
                       ):
        
        """
        Evolution step for `avg` or `min` volume measure type - keep the most successive attempt.
        """
        max_child: GraphObject = GraphObject(identification="decoy", 
                                             _entity="decoy", 
                                             fitness=self.graph_object_default_fitness_value,
                                             )
        max_vol: Union[None, float] = None 
        max_weak_index: Union[None, float] = None
        max_child_fitness: Union[None, float] = None
        
        
        total_cumulative_sum_remaining, indices_to_try_as_weak = self._preprocess_before_evolution_step(population_fitnesses=population_fitnesses)
        
        children = self.generate_children(step_number=step_number, population_fitnesses=population_fitnesses)
        
        found_child_earlier = False
        
        index_of_weak_preprocessed_values = {}
        
        for child in children:
            for index_of_weak in indices_to_try_as_weak:

                if index_of_weak not in index_of_weak_preprocessed_values:
                    index_of_weak_preprocessed_values[index_of_weak] = self._preprocess_before_attempt(index_of_weak=index_of_weak)
                
                min_distance_without_weak, weak_graph_vol_norm = index_of_weak_preprocessed_values[index_of_weak]
                
                child_report = self.try_volume_with_new_child(child=child, 
                                                    current_cum_sum=total_cumulative_sum_remaining - weak_graph_vol_norm + min_distance_without_weak,
                                                    index_of_weak=index_of_weak)
                
                new_vol = child_report["vol"]
                child_fitness = child_report["fitness"]
                
                if max_vol is None or new_vol > max_vol:
                    max_vol = new_vol
                    max_child = child
                    max_weak_index = index_of_weak
                    max_child_fitness = child_fitness
                if max_vol > current_volume:
                    found_child_earlier = True
                    break
                
                self._postprocess_after_attempt(index_of_weak=index_of_weak, weak_graph=self.population[index_of_weak])
                
            
            self.children_checked += 1

            if found_child_earlier:
                break
        
        
            
                 
        current_cumsum = self._postprocess_after_evolution_step(total_cumulative_sum_remaining=total_cumulative_sum_remaining,
                                                                weak_graph_vol_norm=weak_graph_vol_norm,
                                                                max_vol=max_vol,
                                                                )

        return dict(
                current_cumsum=current_cumsum,
                max_vol=max_vol,
                max_weak_index=max_weak_index,
                max_child=max_child,
                max_child_fitness=max_child_fitness,
            )        
    
    @time_wrapper
    def evolve(self):
        """
        Simulate evolution process of population
        """
        
        
        # now self.population is sorted in DESCENDING order according to cumulative distance - this is fitting parameter
        # More fit - closer to the left end of self.population
        
        
        max_volume_population = self.get_graphs_from_population(self.population)
        
        step = 0
        last_cache_purge = 0
        last_checkpoint_save = 0
        # initial step
        with Parallel(**self.workers_options) as workers:
            max_volume = current_volume = self.recompute_and_return_fitness(workers=workers)
            
            self.volumes.append(current_volume)            
            
            while self.children_checked < self.total_attempts:

                
                population_fitnesses: List[float] = [graph_object.fitness for graph_object in self.population]
                new_volume = self.evolution_step_wrapper(population_fitnesses=population_fitnesses, 
                                                            step_number=step, 
                                                            current_volume=current_volume
                                                            )                
                if abs(self.children_checked - last_cache_purge) > 10000:
                    last_cache_purge = self.children_checked
                    self._purge_cache()
                    
                true_vol = self.recompute_and_return_fitness(workers=workers)
                
                self.volumes.append(true_vol)
                self.graphs_generated.append(self.children_checked)
                
                if true_vol > max_volume:
                    max_volume = true_vol
                    max_volume_population = self.get_graphs_from_population(self.population)
                    
                
                if abs(self.children_checked - last_checkpoint_save) > self.checkpoint_interval:  
                    last_checkpoint_save = self.children_checked

                    self.checkpoint(graphs=max_volume_population, name=f"step_{step}_best.pkl")
                    
                    
                step = self.children_checked

                log = f"Children: {step}/{self.total_attempts}\tVolume: {true_vol:.6f}\tDelta: {(true_vol - current_volume):.6f}"
                logger.info(log)

                current_volume = true_vol

                

        self.checkpoint(graphs=max_volume_population, name="final_best.pkl")
        
    def checkpoint(self, graphs: List[Graph], name:str):
        filename = self.checkpoint_dir / name
        save_pickle(obj=graphs, to_filename=filename)
        
class EvolutionalAlgorithm:
    """
    An orchestrator of all combinations of parameters for genetic algorithm. Launches combinations and stores results.
    """
    def __init__(self,
                 initial_graphs: Dict[str, List[GraphObject]],
                 genetic_config: Dict[str, Any],
                 distances_configurations: Dict[Distance_name, List[Parameters]],
                 n_jobs: int,
                 number_of_nodes:int,
                 save_dir: Path or None,
                 orca_path: Path,
                 volume_func_type:str,
                 checkpoint_steps_interval:int=100000,
                ) -> None:
        
        PRECOMPUTE_DICT["GCD"] = partial(PRECOMPUTE_DICT["GCD"], orca_path=orca_path)
        
        self.number_of_nodes = number_of_nodes
        self.initial_graphs = initial_graphs
        
        
        self.parameters =  genetic_config
        logger.info(self.parameters)        
        
        self.checkpoint_steps_interval = checkpoint_steps_interval

        
        self.workers_options = dict(n_jobs=n_jobs, backend="threading", batch_size="auto")
        
        self.save_dir = save_dir
        
        self.distances_names_subnames_functions = []
        for distance_name, params_list in distances_configurations.items():
            for params in params_list:
                self.distances_names_subnames_functions.append([distance_name, 
                                                                get_label(distance_name, params), 
                                                                DISTANCE_TO_FUNC[distance_name]
                                                                ]
                                                               )

        self.metareports = defaultdict(dict)
        self.global_report = None
        self.volume_func_type = volume_func_type
        
    
    @time_wrapper
    def run_every_combination_of_evolutional_algo(self):
        """
        Creates all possible combinations of genetic algorithm parameters, launches evolutional algorithm for each combination and stores results in metareport and .csv tables
        """ 
        parameters_with_names = []
        
        for parameter_name, parameters in self.parameters.items():
            parameters_with_names.append(
                [(parameter_name, param) for param in parameters]
            )
            
        parameters_products = tuple(product(*parameters_with_names))
        
        # columes of future overall report
        COLUMNS = ["run", "distance", "initial_population"] + [params[0][0] for params in parameters_with_names] + ["volume_start", "volume_finish"]
        # entries of this list are datapoints from each run of genetic algorithm
        DATA = []
        
        header = True
        
        metareports_chunks_save_dir: Path = self.save_dir / "metareports_chunks"
        metareports_chunks_save_dir.mkdir(parents=True, exist_ok=True)
        
        runs_checkpoints_dir: Path = self.save_dir / "checkpoints_graphs"
        runs_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        
        report_number = 0
        for distance_name, distance_label, distance_func in tqdm(self.distances_names_subnames_functions,
                                                                 desc="Running genetic algo for each distance"):
            
            for initial_population_label, graphs in self.initial_graphs.items():
                
                for parameters_options in tqdm(
                            parameters_products, 
                            total=len(parameters_products),
                            desc=(desc:=f"Peforming genetic algorithms with different options for {distance_label} and {initial_population_label} population"),
                            ):
                    
                    logger.info(desc)                
                    
                    parameters_dict = dict(parameters_options) | {"initial_graphs": graphs[distance_label].copy()}
                    
                    
                    run_label = get_label(prefix=initial_population_label, items=parameters_dict)
                    
                    logger.info(f"Parameters of current setup: {run_label}")

                    current_run_checkpoint_dir = runs_checkpoints_dir / f"run_{report_number}"
                    current_run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    
                    population = GraphsPopulation(
                                                  number_of_nodes=self.number_of_nodes,
                                                  workers_options=self.workers_options,
                                                  distance_name=distance_name,
                                                  distance_function=distance_func,
                                                  volume_func_type=self.volume_func_type,
                                                  checkpoint_interval=self.checkpoint_steps_interval,
                                                  checkpoint_dir=current_run_checkpoint_dir,
                                                  **parameters_dict)

                    population.evolve()
                    
                    metainfo = population.metainfo
                    volume_first = population.volumes[0]
                    volume_last = population.volumes[-1]
                    
                    self.metareports[distance_label][run_label] = metainfo
                    
                    data_from_run = [report_number, distance_label, initial_population_label] + [param for _, param in parameters_options] + [volume_first, volume_last]
                    
                    if self.save_dir:
                        online_data_chunk = pd.DataFrame(data=[data_from_run], columns=COLUMNS)
                        online_data_chunk.to_csv(self.save_dir / "genetic_runs.csv", mode="a", header=header, index=False)
                        header = False
                        
                        metareport_filename = metareports_chunks_save_dir / f"report_{report_number}.pkl"
                        
                        save_pickle(obj=metainfo, to_filename=metareport_filename)
                    
                    report_number += 1
                    
                    DATA.append(data_from_run)
                    
                    logger.info("Finished with current setup. Moving to the next or exiting if it was the last one.")
        
        # sets overall report to its attribute
        global_data = pd.DataFrame(
                data=DATA,
                columns=COLUMNS,
            ).set_index("run").sort_index()
        
        self.global_report = global_data
        
        
        logger.info(f"Finished all setups. The data of all setup has been collected. Returning to the main module.")