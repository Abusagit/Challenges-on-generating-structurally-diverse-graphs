import re
import pickle
import logging
import pandas as pd
import networkx as nx
import numpy as np
import igraph as ig

from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from functools import partial, reduce
from tqdm import tqdm
from scipy.stats import skew, kurtosis

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS

from utils import get_label, get_graphs_from_metareport, GraphModel, DistanceReport
from utils import Distance_func, Graph_random_model, Distance_name, Parameters, Graphs_dict, Report, Graph
from typing import Dict, List, Tuple, Any, Union
from distances import DISTANCE_TO_FUNC
# from greedy import VOLUME_TO_FUNC


logger = logging.getLogger(__name__)

class SBMNameReplacer:
    # SBM name labels are too long, thus this class tries to shorten it and store full nome for every SBM model after compression
    def __init__(self):
        self.model_to_params = {}
        self.i = 1
        self.pattern = re.compile("SBM_block_sizes_\[(\d+[.,]*\d*,*\s*)+\]_pref_matrix_\[(\[(\d+[.,]*\d*,*\s*)+\],*\s*)+\]")
    
    def __repr__(self):
        return repr(self.model_to_params)
    
    def __getitem__(self, key:str):
        return self.model_to_params[key]
    
    def get(self, key:str, replaced:Any=None):
        return self.model_to_params.get(key, replaced)
    
    def replace_sbm_long_name(self, string_to_be_replaced):
        
        
        try:
            match_start, math_end = self.pattern.search(string_to_be_replaced).span()
        
        
            sbm_class = string_to_be_replaced[match_start:math_end]
            
            if sbm_class not in self.model_to_params:
                self.model_to_params[sbm_class] = f"SBM_{self.i}"
                self.i += 1
            
            replaced_string = self.pattern.sub(self.model_to_params[sbm_class], string_to_be_replaced)
            return replaced_string
                
            
        except AttributeError: # no match
            return string_to_be_replaced


class Analyzer:
    def __init__(self, 
                 greedy_metareport: Dict[str, List[Report]] or None, 
                 distances_generation_report: Dict[str, DistanceReport] or None, 
                 genetic_report: pd.DataFrame or None,
                 data_dir:Path,
                 colors=DEFAULT_PLOTLY_COLORS,
                 ) -> None:
        
        """
        params:
        
        greedy_metareport - metareport of greedy algorithm (can be None if this step ws skipped)
        distances_generation_report - report of analysis of graph random models (can be None if this step ws skipped)
        genetic_report - report of genetic algorithm (can be None if this step ws skipped)

        
        
        data_dir - directory root where of stored results
        colors - color pellete for plots
        
        """
                
        self.sbm_replacer = SBMNameReplacer()
        self.distances_report = distances_generation_report
        self.metareport = greedy_metareport
        self.genetic_report = genetic_report
        
        self.data_dir = data_dir
        
        self.greedy_report_df = self._get_df_from_greedy_report() if greedy_metareport else None
        self.distances_df = self._get_distances_df() if distances_generation_report else None
        
                    
        self.names = list(sorted(set(self.distances_df.index.get_level_values(0)))) if self.distances_report else None # first level of multiindex
        
        self.colors = colors
        
        # basic plot functions
        self.plots = {
            "metrics_plot": self.plot_metrics,
            "greedy_mean_and_variance": self.plot_variance_of_volumes_in_greedy_runs,
            "greedy_graph_distribution_example": partial(self.plot_graphs_distribution, 
                                                        report_dict=self.metareport[list(self.metareport.keys())[0]][0]) if self.metareport else lambda x: None,
            "genetic_trajectories": self.plot_genetic_evolution_in_one_plot,
            "genetic_graphs_properties": self.plot_genetic_graphs_properties,
            
        }
        
        
        
    def _load_pickle_run_by_index(self, index) -> Dict[str, Any]:
        """
        Loads genetic report of a run with assigned index from data_dir
        """
        return pickle.load(file=open(self.data_dir / "metareports_chunks" / f"report_{index}.pkl", "rb"))
    
    def _get_df_from_greedy_report(self) -> pd.DataFrame:
        """
        Transforms greedy report dict structure to pandas DataFrame
        """
        
                
        data = []
        
        for init_type, runs_reports in self.metareport.items():
            
            for report in runs_reports:
                
                for distance_name, distance_report in report.items():
                    if distance_name == "metainfo":
                        continue
                        
                    labels, volume = distance_report["labels"], distance_report["volume"]
                        
                    data.append((distance_name, init_type, volume, labels))
                    
        
        df = pd.DataFrame(data=data, 
                        columns=["Distance name",
                                "Init type",
                                "Volume",
                                "Labels"]).set_index("Distance name").sort_index()
        
        return df
    
    def _get_distances_df(self) -> pd.DataFrame:
        """
        Transforms graph random models report to pandas DataFrame
        """
        data = []
        index = []

        data_columns_names = ["Model label", "Volume"]
        index_names = ["Distance", "Combination size"]
        
        greedy_report = self.metareport[list(self.metareport.keys())[0]][0] if self.metareport else set()

        for distance_name, distance_report in self.distances_report.items():

            if distance_name in greedy_report:
                greedy_vol = greedy_report[distance_name]["volume"]

                data.append(("Greedy", greedy_vol))
                index.append((distance_name, 1))
                
            if isinstance(self.genetic_report, pd.DataFrame) and distance_name in self.genetic_report["distance"].unique():
                
                genetic_vol_start, genetic_vol_finish = self.genetic_report.query("distance == @distance_name").sort_values(
                                                                                                by="volume_finish", ascending=False
                                                                                                ).reset_index().loc[0, ["volume_start", "volume_finish"]]
                
                data.append(("Genetic start", genetic_vol_start))
                index.append((distance_name, 1))
                
                data.append(("Genetic finish", genetic_vol_finish))
                index.append((distance_name, 1))             

            # print(distance_report.report_dict)
            # print(distance_name)
            for combination_size, models_combinations_to_volumes in distance_report.report_dict.items():

                for model_label, vol in models_combinations_to_volumes.items():

                    model_label = self.sbm_replacer.replace_sbm_long_name(model_label)

                    data.append((model_label, vol))

                    index.append((distance_name, combination_size))

        multiindex = pd.MultiIndex.from_tuples(index, names=index_names)
        distances_df = pd.DataFrame(data=data, index=multiindex, columns=data_columns_names)

        return distances_df
    
    def plot_variance_of_volumes_in_greedy_runs(self):
        
        """
        If greedy algorithm is launched multiple times, the plots of mean and std of resulting volume measures can be made
        """
        
        if self.greedy_report_df is None:
            return
        
        mean_std_for_each = self.greedy_report_df[["Volume", "Init type"]].groupby(
                                                                            ["Distance name", "Init type"]
                                                                            ).aggregate(["mean", "std"])
        
        distances_names = list(sorted(set(self.greedy_report_df.index)))
        
        fig = make_subplots(
            cols=1, rows=len(distances_names),
            subplot_titles=distances_names,
            shared_yaxes=True
        )
        
        for row, distance_name in enumerate(distances_names, 1):
            df_distance_view = mean_std_for_each.loc[distance_name]
        
            init_types = []
            means = []
            std = []
            
            for init_type, X in df_distance_view.iterrows():
                init_types.append(init_type)
                means.append(X[0])
                std.append(X[1])
            
            fig.add_trace(go.Scatter(x=init_types, y=means,
                        mode='lines+markers',
                        name=f'{distance_name} mean',
                        line = dict(color='firebrick', width=2, dash="dot")), row=row, col=1,
                    )
            
            fig.add_trace(go.Scatter(x=init_types, y=std,
                        mode='markers',
                        name=f'{distance_name} std',
                        line=dict(color='firebrick', width=2, dash='dash')), row=row, col=1,
                    )
        
        fig.update_layout(height=1000, width=700,
                  title_text=f"Volume mean (dashed) and std (dots) for each init type",
                  showlegend=False)
        
        return fig
    
    def plot_metrics(self, all_sizes=False, height_width=None):
        """
        Accumulates volume measures from all possible experiments and plots them in single plot
        """
        
        if not self.distances_report:
            return
    
        names = self.names
        
        if all_sizes:
            sizes = list(sorted(set(self.distances_df.index.get_level_values(1)))) # get second level of multiindex
        else:
            sizes = [1]

        titles = []
        for name in names:
            for s in sizes:
                titles.append(f"<b>{name}</b>, combination of size {s}")
                
        fig = make_subplots(
            cols=len(sizes), rows=len(names),
            subplot_titles=titles,
            shared_yaxes=True,
#             horizontal_spacing = 0.5,
#             vertical_spacing=1
        )


        
        for row, name in enumerate(names, 1):
            df = self.distances_df.loc[name]


            for col, size in enumerate(sizes, 1):

                data = df.loc[size]

                x = data["Model label"]
                y = data["Volume"]

                if len(data.shape) == 1:
                    x = [x]
                    y = [y]
#                 print(f"{row=} {col=} Done")

                fig.add_trace(go.Box(x=x, y=y),
                              row=row, col=col,
                                )

        if height_width is None:
            height = 400 * len(names)
            width=800 * len(sizes)
            
        else:
            height, width = height_width
            
        fig.update_layout(height=height, width=width,
                          title_text=f"Multiple sizes of models combinations (along columns) with corresponding volumes for each graph distance (along rows)",
                         showlegend=False)

        return fig
    
    
    def plot_graphs_distribution(self, report_dict, histnorm=None):
        
        """
        PLot of distribution of graph random models in the set of graphs chosen by greedy algorithm
        """
        
        subplot_title_pattern = "{} ({})"
        labels_per_distance = {}
        volume_per_distance = {}

        for distance_name, distance_report in report_dict.items():
            labels_per_distance[distance_name] = list(map(self.sbm_replacer.replace_sbm_long_name, distance_report["labels"]))
            volume_per_distance[distance_name] = distance_report["volume"]
        

        titles = [subplot_title_pattern.format(distance, volume_per_distance[distance]) for distance in labels_per_distance.keys()]
        
        fig = make_subplots(
            rows=len(titles),
            cols=1,
            subplot_titles=titles,
        )
        
        
    
        for i, (_, labels) in enumerate(labels_per_distance.items(), 1):


            counts = Counter(labels)
            
            fig.add_trace(
                 go.Histogram(x=sorted(labels, key=lambda x: counts[x], reverse=True), histnorm=histnorm),
                row=i,
                col=1
            )

        
        fig.update_layout(height=1500, width=1000,
                  title_text=f"Histograms of graphs in max diverse set with volume in brackets",
                  showlegend=False)

        return fig
    
    def _get_population_volumes_array(self, indices: List[int], desired_len: int, pad: bool=False) -> np.array:
        """
        Loads genetic trajectories of runs with `indices` described as arrays of volumes at each iteration. Padded to `desired_len` if needed
        """
        array = []
        for index in indices:
            res = self._load_pickle_run_by_index(index=index)["volumes"]
            
            if pad and len(res) < desired_len:
                # continue
                x = desired_len - len(res)
                for _ in range(x):
                    res.append(res[-1])
                    
            array.append(res)
        return np.array(array)
    
    @staticmethod
    def _add_trace(x_value, y_value, mode, line_options, **kwargs) -> go.Scatter:
        return go.Scatter(
                    mode=mode,
                    x=x_value,
                    y=y_value,
                    line=line_options,
                    **kwargs
                )
    
    
    def plot_genetic_evolution_in_one_plot(self, pad=True):
        """
        Plots trajectories of volumes of genetic algorithm
        """
        if isinstance(self.genetic_report, type(None)):
            return
        
        
        distances = self.genetic_report["distance"].unique()
        
        fig = make_subplots(
            rows=1,
            cols=len(distances),
            subplot_titles=distances,
            shared_xaxes=True,
        )
        
        showlegend = True
        for col, distance in enumerate(distances, 1):
            current_distance_genetic_report = self.genetic_report.query("distance == @distance")
            
            
            iterations_nums = self.genetic_report["total_iterations"].unique()
            
            max_iterations = max(list(iterations_nums))
            X = np.arange(max_iterations)
            # print(max_iterations)
            
            traces_to_plot = []
            
            
            for j, total_iterations in enumerate(iterations_nums):
                
                current_distances_genetic_report_for_current_iterstions = current_distance_genetic_report.query(
                    "total_iterations == @total_iterations"
                )
                
                
                initial_populations = current_distances_genetic_report_for_current_iterstions["initial_population"].unique()
                
                
                
                # showlegend = True
                for i, init_population in enumerate(initial_populations):
                    
                    dist_report_iterations_init_population = current_distances_genetic_report_for_current_iterstions.query(
                                                                                        "initial_population == @init_population"
                                                                                    )
                    
                    indices = dist_report_iterations_init_population.index.values
                    
                    
                    volumes_logs = self._get_population_volumes_array(indices=indices, pad=pad, desired_len=total_iterations + 1)
                    
                    # print(indices, volumes_logs)
                    # print(indices, init_population, total_iterations, distance, volumes_logs)
                    means = np.mean(volumes_logs, axis=0)
                    stds = np.std(volumes_logs, axis=0)
                    
                    mean_line_option = dict(color=self.colors[j], width=1.5)
                    mean_std_line_option = dict(color=self.colors[j], width=1.5, dash="dot")
                    
                    y_values_raw = [means, means + stds, means - stds]
                    try:
                        y_values_padded = [np.pad(y, 
                                                pad_width=(0, max_iterations-len(y) + 1), 
                                                mode="constant", 
                                                constant_values=np.nan
                                                ) for y in y_values_raw]
                    except ValueError:
                        y_values_padded = y_values_raw
                        
                    line_options = [mean_line_option, mean_std_line_option, mean_std_line_option]
                    
                    showlegends = [showlegend, False, False]                
                    
                    for y, line_option, legend in zip(y_values_padded, line_options, showlegends):
                        traces_to_plot.append(
                            dict(
                                x_value=X,
                                y_value=y,
                                mode="lines",
                                line_options=line_option,
                                legendgroup=f"group{j}",
                                showlegend=legend,
                                name=f"{total_iterations} iterations"
                                
                            )
                        )
            
            showlegend = False    
            
            for trace_option in traces_to_plot:
                fig.add_trace(
                        self._add_trace(
                            **trace_option
                        ),
                        row=1,
                        col=col,

                    )
            
            
        
        # fig.update_xaxes(type="log", row=row, col=col)            

        fig.update_layout(height=400, 
                        width=600 * len(distances),
                        # title_text="Starting andcd finishing volumes mean (lines) bounded with std ranges (dots) for Greedy population (upper) and ER population (lower)",
                        showlegend=False,
                        legend_title="Different runs",
                        font=dict(size=16)
                        
                        )
        fig.update_xaxes(title_text="Iteration", 
                        #  type='log',
                        )
        fig.update_yaxes(title_text="Volume measure")
        
        return fig
    
    
    
    
    def plot_genetic_graphs_properties(self, additional_graphs_with_labels:Union[None, Dict[str, List[Tuple[List[Graph], str]]]]=None):
        """
        Plots distributions of graph properties of genetic graphs, ER-0.5 graphs and `additional_graphs_with_labels`
        """
        graphs_dict = {}
        
        distances = set()
        for run_index, graph_distance, *_ in self.genetic_report.itertuples():
            distances.add(graph_distance)
            erdos_renyi_graphs = get_graphs_from_metareport(self._load_pickle_run_by_index(index=run_index))
            num_of_graphs = len(erdos_renyi_graphs)
            num_of_nodes = erdos_renyi_graphs[0].number_of_nodes()
            graphs_dict[graph_distance] = [(erdos_renyi_graphs, "genetic")]
        
        
        graphs_genetic_df = create_df_from_run(graphs_and_its_label=graphs_dict)
        
        graphs_characteristics = list(set(list(graphs_genetic_df.columns)) - {"label", "iso", "model", "isomorphic_graphs", "transitivity", 
                                                              "number_of_connected_components", "kurtosis", "skewness",
                                                              "radius", "density", "avg_shortest_path_lentgh",
                                                              }
                              )
        graphs_characteristics.sort()

        characteristic_to_name = {
            "avg_clustering_coefficient": "Avg. clust. coeff.",
            "diameter": "Diameter",
            "radius": "Radius",
            "transitivity": "Transitivity",
            "avg_node_degree" : "Avg. node deg.",
            "number_of_connected_components": "# of CC",
            "total_triangles": "# of \u25B3",
            "avg_shortest_path_lentgh": "Avg. SP length",
            "density": "Density",
            "skewness": "Skewness",
            "kurtosis": "Kurtosis",
            "efficiency": "Efficiency",
        }
        
        er_p_0_5_graphs = [nx.random_graphs.fast_gnp_random_graph(n=num_of_nodes, p=0.5) for _ in range(num_of_graphs)]
        
        random_models_graphs = {}

        for distance in distances:
            random_models_graphs[distance] = [(er_p_0_5_graphs, "ER-0.5")]
            
        
        random_graphs_properties_df = create_df_from_run(graphs_and_its_label=random_models_graphs, isomorphic=False)
        
        graphs_df = pd.concat([random_graphs_properties_df, graphs_genetic_df])
        
        if additional_graphs_with_labels:
            additional_graphs_df = create_df_from_run(graphs_and_its_label=additional_graphs_with_labels, isomorphic=False)
            graphs_df = pd.concat([graphs_df, additional_graphs_df])

        
        distances = list(graphs_df["model"].unique())
        
        labels = list(graphs_df["label"].unique())
        
        titles = []
        
        rows = 0
        
        for label_ in labels:
            if label_ in {"greedy", "genetic"}:
                for distance in distances:
                    titles.extend([f"{label_.capitalize()} ({distance})"] + ["" for _ in range(len(graphs_characteristics) - 1)])
                    rows += 1

            else:
                titles.extend([f"{label_}"] + ["" for _ in range(len(graphs_characteristics) - 1)])
                rows += 1 
                
        fig = make_subplots(
            rows=rows, cols=len(graphs_characteristics),
            subplot_titles=titles,
            shared_xaxes=True,
            # vertical_spacing=0.3,
            horizontal_spacing=0.05,
            # shared_yaxes=True,

        )
        
        
        
        # rows - different runs
        # cols - different graph properties


        for col, graph_property in enumerate(graphs_characteristics, 1):
            
            row = 1
            legend=True
            for label_ in labels:
                
                graph_df_with_label = graphs_df.query("label == @label_")
                
                for graph_distance in distances:
                    
                    distance_property = graph_df_with_label.query("model == @graph_distance")[graph_property].values
                    
                    trace = go.Histogram(
                        x=distance_property,
                        histnorm="probability density",
                        legendgroup=f"group{col}",
                        name=characteristic_to_name[graph_property],
                        marker_color=self.colors[col-1],
                        showlegend=legend,
                    )
                    legend = False
                    fig.add_trace(trace, row=row, col=col)
            
                    row += 1
                    if label_.lower() not in {"greedy", "genetic"}:
                        break
                
                
            
        
        fig.update_layout(height=250 * rows + 20, width=230 * len(graphs_characteristics),
                            font=dict(
                                        # family="Courier New, monospace",
                                        size=16,
                                        color="Black",
                                    )
                        )
                        
                        
        return fig
        
    
    def save_available_plots(self, save_dir:str):
        """
        Tries to draw available plots and save them to `save_dir`/plots dir
        """
        
        plots_dir = Path(save_dir, "plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        for plot_name, plot_func in self.plots.items():
            
            try:
                fig = plot_func()
                fig.write_html(plots_dir / f"{plot_name}.html")
            except BaseException:
                logger.info(f"Fail to plot {plot_name}")
    
def measure_volume_in_single_model(graphs: Dict[Graph_random_model, Dict[str, GraphModel]], 
                                   distance_metric: Distance_func, 
                                   volume_function: Any, 
                                   distance_name: Distance_name, 
                                   n_jobs: int):
    
    """
    
    
    graphs: 
    distance_metric: current distance metric
    volume_function: function for volume calculation
    distance_name: distance_name for cache access
    n_jobs: number of threads
    
    
    return (, )
    """
    
    # defaults
    max_diverse_model = None
    max_volume = float("-inf")
    max_diverse_set = None
    
    model_volume_for_current_distance_metric = {}
    max_volume_models = []        
    
    for model_name, model_configurations in tqdm(graphs.items(), total=len(graphs),
                                                    desc=f"Measuring volumes in single models using distance function {distance_name}"):
        
        overall_index_representations = []
        
        for model_label, model in model_configurations.items():
            
            # cache graph by its model label and index for caching distance pairs!!!!!!!
            graph_labels_representations = [(f"{model_label}_{i}", repres) for i, repres in enumerate(model.precomputed_descriptors.get(distance_name, model.graphs))] # if we have precomputed values for this distance - use them, or use graphs themselves if not
            # breakpoint()
            diversity = volume_function(graph_labels_representations, distance_metric, distance_name, n_jobs)
            
            if diversity > max_volume:
                max_volume = diversity
                max_diverse_model = model_label
                max_diverse_set = graph_labels_representations
            
            model_volume_for_current_distance_metric[model_label] = diversity
                
            overall_index_representations.extend(graph_labels_representations)

        total_diversity = volume_function(overall_index_representations, distance_metric, distance_name, n_jobs)
        
        total_model_label = f"{model_name}_total"

            
        model_volume_for_current_distance_metric[total_model_label] = total_diversity
        
        
        max_volume_models.append((max_diverse_model, max_diverse_set, max_volume))
        
        max_diverse_model = None
        max_volume = float("-inf")
        max_diverse_set = None
    
    return model_volume_for_current_distance_metric, max_volume_models
    

def measure_volume_in_model_combinations(
            max_diverse_model_set: List[Tuple[str, List[List[int]]]],
            n_jobs: int,
            distance_metric: Distance_func,
            distance_name: Distance_name,
            volume_function: Any,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
    
    
    report_dict = defaultdict(dict)
    max_combinations = {}
    
    N = len(max_diverse_model_set) # number of models
    
    for combination_size in tqdm(range(2, N + 1), desc=f"Estimating volume of models combinations with distance function: {distance_name}"): # all combinations of size from 2 to N
        
        combination_of_models = combinations(max_diverse_model_set, combination_size)
        
        max_volume_in_group = float("-inf")
        max_volume_combination = None
        max_volume_set = None
        
        for combination_C_n_k in combination_of_models:
            combination_label = "__".join(sorted(model[0] for model in combination_C_n_k))
            
            # expand all representations to one single list of representations
            representation_of_combination = reduce(lambda x, y: x + y[1], combination_C_n_k, [])
            
            volume = volume_function(representation_of_combination, distance_metric, distance_name, n_jobs)
            
            if volume > max_volume_in_group:
                max_volume_in_group = volume
                max_volume_combination = combination_label
                max_volume_set = representation_of_combination
                
            report_dict[combination_size][combination_label] = volume
        
        max_combinations[combination_size] = combination_label
        
    return report_dict, max_combinations  


# def analyze_graph_distances(graphs: Graphs_dict, 
#                             distances_configurations: Dict[Distance_name, List[Parameters]],
#                             n_jobs: int,
#                             volume_function: str,
#                             ) -> Dict[str, DistanceReport]:
    
#     """
#     Analyzes graphs for each distance configurations
    
#     returns Distance report for each grapg distance configuration
#     """
    
#     volume_function = VOLUME_TO_FUNC[volume_function]
#     distances_dict_report = defaultdict(dict)
    
#     for distance_name, distance_configs in tqdm(distances_configurations.items(), 
#                                                 total=len(distances_configurations),
#                                                 desc="Processing every distance object..."):
        
        
#         # run default and modified configuration
#         for distance_config in distance_configs:            
#             distance_label = get_label(distance_name, distance_config)
            
            
#             # get distance with user-defined parameters
#             distance_metric = partial(DISTANCE_TO_FUNC[distance_name], **distance_config)
            
#             single_model_vol_for_current_distance, max_volume_models = measure_volume_in_single_model(
#                                                                                     graphs=graphs,
#                                                                                     distance_metric=distance_metric, 
#                                                                                     distance_name=distance_label,
#                                                                                     volume_function=volume_function, 
#                                                                                     n_jobs=n_jobs
#                                                                                 )
            
#             model_combinations_C_n_k_volumes, model_combinations_with_max_volume_per_k = measure_volume_in_model_combinations(
#                                                                                     max_diverse_model_set=max_volume_models,
#                                                                                     n_jobs=n_jobs,
#                                                                                     distance_metric=distance_metric,
#                                                                                     volume_function=volume_function,
#                                                                                     distance_name=distance_label,
#                                                                                 )
                        
#             model_combinations_with_max_volume_per_k[1] = max(max_volume_models, key=lambda x: x[2])[0] # get model label with max volume
#             model_combinations_C_n_k_volumes[1] = single_model_vol_for_current_distance
            
            
#             distances_dict_report[distance_label] = DistanceReport(distance_label=distance_label,
#                                                                    model_combination_C_n_k_to_volume=model_combinations_C_n_k_volumes,
#                                                                    model_combinations_with_max_volumes=model_combinations_with_max_volume_per_k,
#                                                                    )
            
#     return distances_dict_report



def return_unique_graphs(graphs: List[nx.classes.graph.Graph]):
    
    unique_graphs = set()
    
    n = len(graphs)
    for i, graph_i in enumerate(graphs):
        
        for j in range(n):
            if i == j:
                continue
            
            graph_j = graphs[j]
            
            if nx.is_isomorphic(graph_i, graph_j):
                if graph_j not in unique_graphs:
                    unique_graphs.add(graph_i)
                break
        else:
            unique_graphs.add(graph_i)
    
    logger.info(f"Found {len(unique_graphs)} unique graphs")
    return list(unique_graphs)

def count_isomorphic_graphs(graphs: List[nx.classes.graph.Graph]):
    
    pairs = combinations(graphs, 2)
    
    N = len(graphs)
    N *= (N - 1) / 2
    
    isomorphic_pairs = 0
    isomorphic_set = set()
    for g1, g2 in tqdm(pairs, total=N):
        if nx.is_isomorphic(g1, g2):
            isomorphic_set.add(g1)
            isomorphic_set.add(g2)
            isomorphic_pairs += 1
    
    return isomorphic_pairs, list(isomorphic_set)

def count_isomorphic_graphs_for_graph(g, graphs: List[nx.classes.graph.Graph]):
    
    isomorphic_graphs = 0
    
    for g_1 in graphs:
        isomorphic_graphs += nx.is_isomorphic(g, g_1)
    
    return isomorphic_graphs - 1

def get_avg_degree(g):
    return np.mean([d for _, d in g.degree()])


def _compute_shortest_paths_vector(G):
    
    NUM_OF_NODES = G.vcount()
    
    sp_vector = np.zeros(NUM_OF_NODES)
    
    for v in range(NUM_OF_NODES):
        shortest_paths = G.get_all_shortest_paths(v)
        
        
        paths = set()
        for path in shortest_paths:
            path_len = len(path) - 1
            
            if (start:=path[0], finish:=path[-1]) in paths:
                continue
            
            paths.add((start, finish))
            
            sp_vector[path_len] += 1
    
    # print(sp_vector)
    
    return sp_vector

def _metrics_for_graph(G, shortest_paths=False):
    G_1 = G
    G = ig.Graph.Adjacency((nx.to_numpy_matrix(G) > 0).tolist())
    N = G.vcount()
    
    
    graph_properties = {}


    graph_properties["radius"] = G.radius()
    graph_properties["diameter"] = G.diameter()
    graph_properties["density"] = G.density()
    graph_properties["efficiency"] = nx.global_efficiency(G_1)
    # graph_parameters["assortativity"] = G.assortativity()
    
    
    graph_properties["transitivity"] = G.transitivity_undirected()
    graph_properties["total_triangles"] = np.array(list(nx.triangles(G_1).values())).sum()

    graph_properties["avg_clustering_coefficient"] = nx.average_clustering(G_1)
    
    graph_properties["number_of_connected_components"] = nx.number_connected_components(G_1)

    # graph_properties["degree_sequence"] = np.array(G_1.degree())
    
    # graph_properties["local_clustering_coefficient"] = nx.clustering(G_1)
    # graph_properties["eccentricity"] = nx.eccentricity(G_1)
    graph_properties["avg_node_degree"] = get_avg_degree(G_1)
    
    degree_sequence = [d for _, d in G_1.degree()]
     
    graph_properties["skewness"] = skew(degree_sequence)
    graph_properties["kurtosis"] = kurtosis(degree_sequence)
    
    if not shortest_paths:
        return graph_properties
    
    sp = _compute_shortest_paths_vector(G)
    # graph_properties["shortest_paths_lengths"] = sp
    graph_properties["avg_shortest_path_lentgh"] = (sp @ np.arange(N)) / N / (N - 1)
   

    return graph_properties


def create_df_from_run(graphs_and_its_label: Dict[str, 
                                                  List[Tuple[List[nx.classes.graph.Graph], str]]], isomorphic=True, shortest_paths=True):
    
    """
    Creates dataframe for given graphs and labels
    """
    
    data = []
    
    for model in graphs_and_its_label:
        model_dict = {"model": model}
        
        for graphs, label in graphs_and_its_label[model]:
            
            isomorphic_pairs = count_isomorphic_graphs(graphs) if isomorphic else np.nan
            
                        
            label_dict = {"label": label, "iso": isomorphic_pairs} | model_dict
            
            
            for graph in tqdm(graphs):
                graph_properties = _metrics_for_graph(graph, shortest_paths=shortest_paths) | label_dict | {"isomorphic_graphs": count_isomorphic_graphs_for_graph(graph, graphs) if isomorphic else np.nan}
                
                data.append(graph_properties)
            
            
            
    
    df = pd.DataFrame(data)
    
    return df


def get_greedy_random_genetic_reports(directory: Path, finished_genetic=True):
    
    try:
        greedy = pickle.load(open(directory / "greedy_graphs_generation_report.pkl", "rb"))
    except Exception:
        greedy = None
    try:
        distances_report = pickle.load(open(directory / "random_graphs_distances_report.pkl", "rb"))
    except Exception:
        distances_report = None
        
    if finished_genetic:
        genetic_report_path = directory / "genetic_runs_finished.csv"
    else:
        genetic_report_path = directory / "genetic_runs.csv"
        
    genetic_report = pd.read_csv(genetic_report_path)
    
    
    return greedy, distances_report, genetic_report


def get_edge_trace(graph: nx.classes.graph.Graph, graph_layout: dict, legend=True):
    edge_x = []
    edge_y = []
    
    for edge in graph.edges():
        x0, y0 = graph_layout[edge[0]]
        x1, y1 = graph_layout[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=legend,
        )
    
    return edge_trace

def get_node_trace(graph: nx.classes.graph.Graph, graph_layout: dict, legend=True):
    node_x = []
    node_y = []
    for node, position in graph_layout.items():
        x, y = position
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=legend,
            # colorscale options
            #'Greys' | 'YlGnBu'    | 'Greens' | 'YlOrRd'   | 'Bluered'  | 'RdBu' |
            #'Reds'  | 'Blues'     | 'Picnic' | 'Rainbow'  | 'Portland' | 'Jet'  |
            #'Hot'   | 'Blackbody' | 'Earth'  | 'Electric' | 'Viridis'  |        |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            cmin=0,
            cmax=graph.number_of_nodes() - 1,
            line_width=2))


    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'# of connections: '+ str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    node_trace.showlegend = legend
    
    return node_trace


def get_edge_node_traces(G: nx.classes.graph.Graph, layout:str="spring", legend=True):
    if layout == "spring":
        G_layout = nx.spring_layout(G)
    else:
        G_layout = nx.spectral_layout(G)
    
    edge_trace = get_edge_trace(graph=G, graph_layout=G_layout, legend=legend)
    node_trace = get_node_trace(graph=G, graph_layout=G_layout, legend=legend)
    
    return edge_trace, node_trace  

def draw_graph(G: nx.classes.graph.Graph, title="", layout:str="spring", legend=True) -> go.Figure:
    edge_trace, node_trace = get_edge_node_traces(G, layout=layout, legend=legend)
    
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=f'<br>{title}</br>',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=30,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    fig.update_layout(
        height=300,
        width=300
    )
    return fig

def draw_graphs(graphs_list, graphs_number=100, rows=None, cols=None, legend=False):
    if rows is None or cols is None:
        rows = cols = int(np.ceil(np.sqrt(graphs_number)))        
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        vertical_spacing=0.01,
        horizontal_spacing=0.01,
        shared_xaxes=True,
        shared_yaxes=True,
    )


    
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            
            graph = graphs_list[rows * (row - 1) + col - 1]
            
            edges, nodes = get_edge_node_traces(graph, legend=legend)
            legend=False
            fig.add_trace(edges, row=row, col=col)
            fig.add_trace(nodes, row=row, col=col)
            
            
    fig.update_layout(height=250 * rows, 
                    width=250  * cols,
                    #   title_text= "Graphs",
                    showlegend=False,
                    coloraxis=dict(cmax=16, cmin=0),
                    font=dict(size=16)
                    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    return fig