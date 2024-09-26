import sys
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import torch

from tqdm import trange

from src.diffusion_model_discrete import DiscreteDenoisingDiffusion

sys.path.append("../graph_diversity_problems/")

import argparse
import random
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


from base import GraphObject
from distances import DISTANCE_TO_FUNC, ProgressParallel
from generation import get_initial_graphs
from joblib import Parallel, delayed
from tqdm import trange
from utils import read_pickle, save_pickle

NUM_PROCESSES = 12


def generate_graphs_from_model(model, number_of_graphs, batch_size=32, batch_upper_limit=5000):
    
    def get_graphs_adjacencies_from_model(number_of_graphs):
        graphs_node_types_adj = model.sample_batch(
                                        batch_size=number_of_graphs,
                                        batch_id=0,
                                        keep_chain=1,
                                        number_chain_steps=1,
                                        save_final=0
                                    )
        
        adjs = [g[1].numpy() for g in graphs_node_types_adj]
        
        return adjs
    
    number_of_batches = number_of_graphs // batch_size
    graphs_w_o_batches = number_of_graphs % batch_size
    
    graphs_adjacencies = []
    
    for batch in trange(number_of_batches, desc="Sampling by batches with size {}".format(batch_size),
                        total=number_of_batches):
        graphs_adjacencies.extend(
            get_graphs_adjacencies_from_model(number_of_graphs=batch_size)
        )
        
        print(f"{batch+1}/{number_of_batches}")
    else:
        graphs_adjacencies.extend(
            get_graphs_adjacencies_from_model(number_of_graphs=graphs_w_o_batches) if graphs_w_o_batches > 0 else []
        )
        print(f"Extra {graphs_w_o_batches} graphs")
    
    return graphs_adjacencies


def get_graph_objects(adjacencies, distance):
    
    config = {
        "initial_graphs": "user",
    }
    
    graph_with_computed_descriptors = get_initial_graphs(
                                            config=config,
                                            threads=12, 
                                            distances_set={distance}, 
                                            samples=None, 
                                            nodes_number=16, 
                                            orca_path="../graph_diversity_problems/orca/", 
                                            equal_sizes=True, 
                                            maybe_ready_graphs=adjacencies,
                                            greedy_graphs_objects_per_distance=None,
                                        )["user"]
                                            
    
    result_dict = {}
    
    for distance_, graph_label_entity in graph_with_computed_descriptors.items():
        graph_objects = [
            GraphObject(
                _entity=e, identification=i, _graph=g
            ) for g, i, e in graph_label_entity
        ]
        
        result_dict[distance_] = graph_objects
        
    
    return result_dict

def count_pairwise_energy(graphs: List[GraphObject], distance_function:Callable[[Any, Any], float]):
    i_j_indices = combinations(graphs, 2)
    
    distances = ProgressParallel(n_jobs=NUM_PROCESSES)(
        delayed(
            distance_function
        )(e_1, e_2) for e_1, e_2 in i_j_indices
    )
    
    distances = np.array(distances)
    return distances, distances.mean()

def energy_distance(x: GraphObject, y: GraphObject, distance_name:str):
    return 1 / (DISTANCE_TO_FUNC[distance_name](x.entity, y.entity) + 1e-6)



def sample_greedy_from_graphobjects_of_certain_distance(graph_objects:List[GraphObject],
                                                        distance_function:Callable[[Any, Any], float],
                                                        final_set_size:int=100,
                                                        super_greedy=False,
                                                        ):    
    
    
    if not super_greedy:
        N = len(graph_objects)
        competitors_per_sample = N // (final_set_size - 1)
        
        # random_permutation
        indices = np.random.permutation(range(N))
        
        
        offset = 2
        resulting_set: List[GraphObject] = [graph_objects[random.choice(indices[:offset])]]
        
        with Parallel(n_jobs=12) as workers:
            for i in range(final_set_size - 1):
                
                candidate_indices = indices[offset + competitors_per_sample * i : offset + competitors_per_sample * (i + 1)]
                candidates = [graph_objects[k] for k in candidate_indices]
                
                
                distances = np.array(workers(
                    delayed(
                    distance_function 
                    )(already_chosen_graph, candidate) for candidate in candidates for already_chosen_graph in resulting_set
                ))
                
                distances = -1.0 * distances.reshape(len(candidate_indices), -1)
                
                fitnesses = distances.sum(1)
                
                max_fitness_index = fitnesses.argmax()
                
                
                winner = candidates[max_fitness_index]
                
                resulting_set.append(winner)
    else:
        N = len(graph_objects)
        
        # random_permutation
        indices = np.random.permutation(range(N))
        
        graphs = [graph_objects[i] for i in indices]
        offset = 2
        resulting_set: List[GraphObject] = [graphs[0]]
        
        graphs = graphs[1:]
        
        fitnesses = np.zeros(len(graphs))
        
        with Parallel(n_jobs=12) as workers:
            for i in trange(final_set_size - 1):
                
                
                distances = np.array(workers(
                    delayed(
                    distance_function 
                    )(resulting_set[-1], candidate) for candidate in graphs
                ))
                
                distances = distances.reshape(len(graphs), -1)
                
                fitnesses += distances.sum(1)
                
                max_fitness_index = fitnesses.argmin()
                
                winner = graphs[max_fitness_index]
                fitnesses[max_fitness_index] += 1e5
                
                resulting_set.append(winner)


    return resulting_set


def generate_greedy_graphs_for_generated_set(graph_objects_dict:Dict[str, GraphObject],
                                             greedy_set_size:int,
                                             number_of_repeats:int=5,
                                             super_greedy:bool=False,
                                             ):
    
    table = defaultdict(list)
    final_graphs = {}
    M = number_of_repeats

    for distance_name, graph_objects_list in graph_objects_dict.items():
        distance_func = partial(energy_distance, distance_name=distance_name)
        
        max_fitness = -1
        for i in range(M):
            greedy_chosen_graphs = sample_greedy_from_graphobjects_of_certain_distance(graph_objects_list, 
                                                                                        distance_function=distance_func,
                                                                                        super_greedy=super_greedy,
                                                                                        final_set_size=greedy_set_size,
                                                                                        )
            
            distances, fitness = count_pairwise_energy(greedy_chosen_graphs, distance_func)
            
            
            if greedy_set_size > 100:
                _, fitness_100 = count_pairwise_energy(greedy_chosen_graphs[:100], distance_func)
                
                print(f"Fitness of 100 graphs: {fitness_100}")
            
            if fitness > max_fitness:
                final_graphs[distance_name] = [g.graph for g in greedy_chosen_graphs]
                max_fitness = fitness
                
            
            table[distance_name].append(fitness)
            print(f"{distance_name} - {i+1}/{M}")
            
        
        print(f"{distance_name} - done")
        
    overall_table = pd.DataFrame.from_dict(table)


    cols = ["fitness", "distance"]
    data = []
    for d, array in table.items():
        for f in array:
            data.append([f, d])
            
            
    df = pd.DataFrame(data, columns=cols)
    distance_fitness_average_std = df.groupby("distance").aggregate(["mean", "std"])
    
    return final_graphs, overall_table, distance_fitness_average_std


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-w", type=str, default=None)
    parser.add_argument("--generate", "-g", type=int, default=1_000_000)
    parser.add_argument("-b", "--batch_size", type=int, default=5_000)
    parser.add_argument("--graphs", type=Path, default=None)
    parser.add_argument("-o", "--outdir", type=Path, required=True)
    parser.add_argument("--greedy_size", type=int, default=1000)
    parser.add_argument("-d", "--graph_distance", type=str, default="GCD")
    args = parser.parse_args()
    
    print(args)
    outdir: Path = args.outdir
    outdir.mkdir(exist_ok=True, parents=True)
    
    
    model_graphs = []
    if args.weights is not None:
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(args.weights)
        model.visualization_tools = None
        
        print(model)
        print("Sampling graphs from model")
        
        model_graphs = generate_graphs_from_model(model=model, 
                                                  number_of_graphs=args.generate,
                                                  batch_size=args.batch_size,
                                                  )

        model = model.cpu()
        del model
        torch.cuda.empty_cache()
        
    if args.graphs is not None:        
        other_graphs = read_pickle(args.graphs)
        model_graphs += other_graphs
        
        
    if len(model_graphs) == 0:
        raise ValueError("Either graphs or model weights should be specified, but none of them were given.")
    

    
    save_pickle(obj=model_graphs, 
                to_filename=outdir / "generated_graphs.pkl")
        
    
    generated_graph_objects = get_graph_objects(model_graphs, 
                                                distance=args.graph_distance)


    print("Greedy generation!")
    # final_graphs, overall_table, distance_fitness_average_std = generate_greedy_graphs_for_generated_set(
    #     graph_objects_dict=generated_graph_objects,
    #     greedy_set_size=args.greedy_size,
    #     number_of_repeats=5,
    #     super_greedy=False,
    # )

    # print(overall_table, distance_fitness_average_std, sep="\n===========\n")
    
    # overall_table.to_csv(outdir / "overall_table_SUB_greedy.csv")
    # distance_fitness_average_std.to_csv(outdir / "distances_fitnesses_SUB_greedy.csv")
    # save_pickle(final_graphs, outdir / "final_graphs_SUB_greedy.pkl")


    final_graphs, overall_table_super_greedy, distance_fitness_average_std_super_greedy = generate_greedy_graphs_for_generated_set(
        graph_objects_dict=generated_graph_objects,
        greedy_set_size=args.greedy_size,
        number_of_repeats=1,
        super_greedy=True,
    )

    print(overall_table_super_greedy, distance_fitness_average_std_super_greedy, sep="\n===========\n")
        
    overall_table_super_greedy.to_csv(outdir / "overall_table_greedy.csv")
    distance_fitness_average_std_super_greedy.to_csv(outdir / "distances_fitnesses_greedy.csv")
    
    np.save(outdir / "final_graphs_greedy", final_graphs[args.graph_distance])