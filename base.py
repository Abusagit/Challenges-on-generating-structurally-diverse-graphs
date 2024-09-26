from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, MutableSet, Callable, Iterable
from utils import Distance_func, Distance_name, check_for_implementation
from distances import PRECOMPUTE_DICT
from collections import defaultdict
import os

import logging

from itertools import combinations

from joblib import Parallel, delayed

import numpy as np
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

@dataclass(order=True, eq=True)
class GraphObject:
    
    """
    Basic functional for graph in evolutional algorithm, contains:
    
    graph: adjacency matrix
    entity: precomputed object needed for computing graph distance faster
    identification: unique label of graph, comprised from condition it was created, creates unique hash value for each instance
    fitness: fitness value of an instance
    """
    
    
    _entity: Any = field(hash=False, repr=False, compare=False)
    identification: str = field(hash=True, repr=True, compare=False)
    fitness: float = field(default=0.0, hash=False, repr=True, compare=True) # will be changed according to volume function type
    
    _graph_path: Path = field(hash=False, repr=False, compare=False, default=Path("NULL"))

    _prev_fitness: float = field(default=0.0, hash=False, repr=False, compare=False)
    
    id_hash: int= field(default=0, hash=False, repr=False, compare=True) # used for stable sorting only
    distances_to_population: Union[List[float], None] = field(default=None, hash=False, repr=False, compare=False)
    
    graph_filepath: Path = field(hash=False, repr=True, compare=False, default=Path("NULL"))
    entity_filepath: Path = field(hash=False, repr=True, compare=False, default=Path("NULL"))

    _store_graph_on_disk: bool = field(hash=False, repr=False, compare=False, default=False)
    _graph: Any = field(hash=False, compare=False, repr=False, default=None)
    
    def __post_init__(self):
        self.id_hash = hash(self.identification)
        self.graph_filepath = Path(self._graph_path) / f"{self.identification}.npy"
        
        if isinstance(self._entity, str):
            self.entity_filepath = Path(self._entity)
        
    def __hash__(self) -> int:
        return self.id_hash
    @property
    def graph(self) -> np.ndarray:
        if self._store_graph_on_disk:
            return np.load(file=self.graph_filepath)
        else:
            return self._graph
    
    @graph.setter
    def graph(self, adjacency_matrix: np.ndarray) -> None:
        if self._store_graph_on_disk:
            np.save(file=self.graph_filepath, arr=adjacency_matrix)
        else:
            self._graph = adjacency_matrix
            
    @property
    def entity(self) -> Any:
        if self._store_graph_on_disk and isinstance(self._entity, str):
            loaded_entity = np.load(file=self.entity_filepath)
            self._entity = loaded_entity
            return loaded_entity
        else:
            return self._entity
    
    @entity.setter
    def entity(self, entity: Any) -> None:
        if self._store_graph_on_disk and isinstance(self._entity, str):
            np.save(file=self.entity_filepath, arr=entity)
        else:
            self._entity = entity
            
    def __del__(self):
        # delete a graph from the memory storage
        
        try:
            os.remove(self.graph_filepath)
            os.remove(self.entity_filepath)
        except FileNotFoundError:
            pass

class DiversityBaseClass:
    
    def __init__(self,
                 initial_population: List[Tuple[np.ndarray, str, Any]],
                 distance_name: Distance_name,
                 distance_function: Distance_func,
                 volume_func_type: str,
                 MODE:str,

                 total_attempts: int=0,
                 max_failed_attempts: int=0,
                 checkpoint_interval: Union[int, None]=None,
                 checkpoint_dir: Path=Path.cwd(),
                 number_of_nodes: Union[int, None]=None,
                 number_of_graphs: Union[int, None]=None,
                 *args,
                 **kwargs,
                 ) -> None:
        check_for_implementation(volume_func_type=volume_func_type)

        self.graph_object_default_fitness_values = {
            "avg": 1e-6,
            "min": np.inf,
            "energy": 1e-6,
        }
        
        self.graph_object_default_fitness_value = self.graph_object_default_fitness_values[volume_func_type]
        
        
        self.number_of_nodes = number_of_nodes
        self.number_of_graphs = number_of_graphs
        self.total_attempts = total_attempts
        self.max_failed_attempts = max_failed_attempts
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        
        
        self.precompute_entity_func = PRECOMPUTE_DICT[distance_name]
        
        self.graphs_dir = self.checkpoint_dir / "graphs"
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        
        self.MODE = MODE
        
        self.save_graphs_on_a_disk = True if MODE == "greedy" else False
        
        self.distance_name = distance_name
        


        self.volume_func_type = volume_func_type
        
        
        

        if self.volume_func_type == "energy":
            self.distance_function: Callable[[GraphObject, GraphObject], float] = lambda G_1, G_2:  1 / (distance_function(G_1, G_2) + 1e-6) # epsilon for numerical stability
        else:
            self.distance_function: Callable[[GraphObject, GraphObject], float] = distance_function
        
        
        self.volumes = []
                # assign cache
        self.cache: Dict[GraphObject, Dict[GraphObject, float]] = defaultdict(lambda : defaultdict(float))
        
        
        self.population = [self.get_graph_object_from_graph_and_id(graph=adjacency,
                                                                    identification=label,
                                                                    first_init_of_population=True,
                                                                    graph_entity=descriptor,
                                                                    ) for adjacency, label, descriptor in initial_population
                           ]
        self.guys_achieving_minimal_distance: MutableSet[int] = set()

        # for entry in self.population:
        #     entry.fitness = self.graph_object_default_fitness_value
        #     entry.distances_to_population = [self.graph_object_default_fitness_value for _ in self.population]
    
    @property    
    def overall_pairs_doubled(self) -> float:
        return len(self.population) * (len(self.population) - 1)
    @property
    def N(self) -> int:
        return len(self.population)

    
    
    def _graph_distance(self, G_1: GraphObject, G_2: GraphObject, save_in_cache: bool=True) -> float:
        """
        hashable distance computation
        """
        
        if self.cache.get(G_2) and (dist:=self.cache.get(G_2).get(G_1)):
            return dist
        if self.cache.get(G_1) and (dist:=self.cache.get(G_1).get(G_2)):
            return dist
        
        dist = self.distance_function(G_1.entity, G_2.entity)
        
        if save_in_cache:
            self.cache[G_1][G_2] = self.cache[G_2][G_1] = dist

        return dist
    def get_list_of_fitnesses_and_probabilities(self):
        fitnesses = np.array([g.fitness for g in self.population])
        
        return fitnesses, fitnesses / np.sum(fitnesses)
    
    
    def update_fitness(
                self, 
                graph_index_1:int, 
                graph_index_2:int, 
                multiplier:int=1, 
                new_graph:Union[GraphObject, None]=None,
            ) -> Tuple[int, int, float]:
        """
        Updates fitness of graphs at indices `graph_index_1` and `graph_index_2` in self.population.
        Procedure varies from distance to distance
        """
        def update_pair(graph_1: GraphObject, graph_2: GraphObject, distance):
            if self.volume_func_type == "avg" or self.volume_func_type == "energy":
                graph_1.fitness += multiplier * distance
                graph_2.fitness += multiplier * distance
                
            elif self.volume_func_type == "min":
                graph_1.fitness = min(distance, graph_1.fitness)
                graph_2.fitness = min(distance, graph_2.fitness)
                
        G_1 = self.population[graph_index_1]
        G_2 = self.population[graph_index_2]

        dist_1 = self._graph_distance(G_1, G_2)
        
        update_pair(graph_1=G_1, 
                    graph_2=G_2, 
                    distance=dist_1,
                    )

        if new_graph:
            dist_2 = self._graph_distance(new_graph, G_2)
            update_pair(graph_1=new_graph, 
                        graph_2=G_2, 
                        distance=dist_2,
                        )
            return graph_index_1, graph_index_2, dist_2
            
        return graph_index_1, graph_index_2, dist_1


    def compute_volume_easy(self):
        
        if self.volume_func_type in "avg":
            return sum([x.fitness for x in self.population]) / self.overall_pairs_doubled
        
        if self.volume_func_type == "min":
            return self.population[-1].fitness
                
        if self.volume_func_type == "energy":
            return -1.0 * sum([x.fitness for x in self.population]) / self.overall_pairs_doubled
            
        raise NotImplementedError

    def recompute_and_return_fitness(self,  workers: Parallel) -> float: 
        """Computes for each GraphObject in self.population its cumulative distance"""

        for graph in self.population:
            graph.fitness = self.graph_object_default_fitness_value
            graph.distances_to_population = None
        
        # look for the pairs achieving min_distance        
        graph_pairs_indices = combinations(range(len(self.population)), 2)
        i_j_distances = workers(delayed(self.update_fitness)(i, j) for i, j in graph_pairs_indices)
        
        
        if self.volume_func_type == "min" and self.MODE == "genetic":
            min_distance = np.inf
            min_graphs: MutableSet[GraphObject] = set()
            for i, j, dist in i_j_distances:
                if dist <= min_distance:
                    if dist < min_distance:
                        min_graphs.clear()
                        min_distance = dist
                        
                    min_graphs.add(self.population[i])
                    min_graphs.add(self.population[j])
            
            min_graphs: Iterable[GraphObject] = sorted(min_graphs, reverse=True)
        
        #self.population.sort(reverse=self.volume_func_type != "energy") # if energy, then sort in ascending order - lower energy is better

            self.guys_achieving_minimal_distance.clear()
            
            index = 0
            for min_graph in min_graphs:
                # print(min_graph)
                # breakpoint()
                while index < self.N:
                    # breakpoint()
                    if min_graph.identification == self.population[index].identification:
                        # breakpoint()
                        self.guys_achieving_minimal_distance.add(index)
                        break
                    
                    index += 1
            del min_graphs
                
        return self.compute_volume_easy()


    # TODO
    def get_graph_object_from_graph_and_id(self, graph: np.ndarray or str, 
                                           identification: str, 
                                           first_init_of_population: bool=False,
                                           graph_entity: Union[Any, None]=None,
                                           ) -> GraphObject:
        """
        Wrapper for GraphObject creation and code decomposition
        """
        
        if isinstance(graph, str):
            if graph_entity is None:
                graph_entity = self.precompute_entity_func(np.load(graph, allow_pickle=True))
                
            child = GraphObject(
                            _graph_path=graph,
                            identification=identification, 
                            _entity=graph_entity,
                            fitness=self.graph_object_default_fitness_value,
                            _store_graph_on_disk=True,
                            )
            child.graph_filepath = graph # for correct load
            
            
        else:
            
            entity = graph_entity if graph_entity is not None else self.precompute_entity_func(graph)
            child = GraphObject(
                            _graph_path=self.graphs_dir,
                            identification=identification, 
                            _entity=entity,
                            fitness=self.graph_object_default_fitness_value,
                            _store_graph_on_disk=self.save_graphs_on_a_disk,
                            )

            child.graph = graph
            child.entity = entity
        
        if not first_init_of_population:
            distances_to_population = np.array([self._graph_distance(child, y) for y in self.population])
            
            child.distances_to_population = distances_to_population
            
            if self.volume_func_type == "avg" or self.volume_func_type == "energy":
                child.fitness = np.sum(distances_to_population)
                
            elif self.volume_func_type == "min":
                child.fitness = np.min(distances_to_population)
            
        return child
        
    def get_graphs(self):
        return [g.graph for g in self.population]

    def try_volume_with_new_child(self, child: GraphObject, current_cum_sum: float, index_of_weak: int) -> Dict[str, float]:
        
        """
        Function estimates the volume after replacinf deleted graph with new `child`.
        
        `current_cum_sum` - correction of current volume measure
        """
        
        cursed_distance = child.distances_to_population[index_of_weak]
        child.distances_to_population[index_of_weak] = self.graph_object_default_fitness_value
        
        dist_to_weak = self._graph_distance(child, self.population[index_of_weak])
        
        if self.volume_func_type == "avg" or self.volume_func_type == "energy":

            fitness = child.fitness - dist_to_weak
            vol = (2 * fitness + current_cum_sum) / self.overall_pairs_doubled # type: ignore
            
            if self.volume_func_type == "energy":
                vol *= -1
            
        elif self.volume_func_type == "min":
            
            if child.fitness <= dist_to_weak:
                
                fitness = child.fitness
            else:
                
                fitness = np.min(child.distances_to_population)

            vol = min(fitness, current_cum_sum)
            
        child.distances_to_population[index_of_weak] = cursed_distance
        
        return dict(
            vol=vol,
            fitness=fitness,
        )

    def _clear_graph_cache(self, index:int) -> None:
        """
        Code decomposition for cleaning cache after deletion of graph from population
        """
        weak_graph = self.population[index]
        del self.cache[weak_graph]
        
        for remaining_graph in self.population:
            try:
                del self.cache[remaining_graph][weak_graph]
            except KeyError:
                pass

    def update_population(self, weak_graph_index:int, new_graph:GraphObject) -> None:
        """
        Pupulation fitness update and resorting by corresponding fitnesses if needed.

        """
        
        # self.parallel_workers(
        #     delayed(self.update_fitness)(weak_graph_index, g_index, -1, new_graph) for g_index in range(self.N)
        # )
        
        self._clear_graph_cache(index=weak_graph_index)
        self.population[weak_graph_index] = new_graph
    
    def _purge_cache(self):

         logger.debug("Purging the cache. Might take a while...")
         graphs_identities: MutableSet[str] = set([g.identification for g in self.population])

         keys = list(self.cache.keys())

         for key in keys:
             if key not in graphs_identities:
                 del self.cache[key]

                 for remaining_graph in self.population:
                     try:
                         del self.cache[remaining_graph][key]
                     except KeyError:
                         pass
         logger.debug("Purging complete. Moving on.")
         
    @staticmethod
    def get_graphs_from_population(population: List[GraphObject]) -> List[np.ndarray]:
        return [x.graph for x in population]
    