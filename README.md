# Challenges on generating structurally diverse graphs
## Description
This repository contains code used in the article "Challenges of Generating Structurally Diverse Graphs" as well as notebooks for analysis and implemented greedy and genetic algorithms.

## Installation

To install all dependencies, please clone this repository and start dedicated Python environment with `python>=3.9`


To install all dependencies, please run the following command:

```{bash}
conda create -n graph_diversities python==3.10
pip install -U numpy scipy pandas scikit-learn tqdm plotly kaleido networkx igraph netrd netlsd joblib 
```

### Installing ORCA orbit counting algorithm

To install orca algorithm, compile the file `orca/orca.cpp`:

```{bash}
g++ -o orca/orca orca/orca.cpp
```

Compiled file will be stored in `orca/orca` path, which is default location for all created configs. Feel free to modify destination and change it config files accordingly.

## Run experiments

### Run pipeline

To start pipeline, you can run the following command:

```{bash}
python pipeline.py  --threads <t> \
                    --outdir <your outdir> \
                    --mode <genetic\greedy\random\all> \ 
                    --config <your config location> 
                    --graphs <if you want to start your algorithm from your graphs. Should be serialized list of networkx graphs>]

```

Optional arguments:

- `analyze` - pass this argumant to get some simple graphs about your run, s.t. plots of obtained diversities of models, distribution of models inside greedy algorithm
- `force` - pass this argument if you want to override directory from `outdir` parameter (_be careful!_)

### Run IGGM (Iterative Graph Generative Modelling)

To run IGGM on your input graphs, you first need proceed to `DiGress` directory and follow `README` for the installation instructions. Then you can configure your training procedure via modifying `user` mode of data. In DiGress you can modify configs to change the behaviour of the neural network.

Then, you can use the following command:

```{bash}
conda activate digress # activate digress envoronment

python DiGress/run_digress_sequentially.py      --graphs_limit <maximum number of generated graphs> \
                                                --batch_size <batch size for the graphs generation, NOT FOR TRAINING>
                                                --outdir <your outdir> \
                                                --graphs <your input graphs> \ 
                                                --num_iterations <number of neural network training iterations> \ 
                                                --graph_distance <GCD/Portrait/netLSD_heat/netLSD_wave> \ 
                                    

```

### Config options

The example of config is located in `configs/` directory. Config is a file with broader instructions for behavior of random graphs generators, graph distances, greedy algorithm and genetic algorithm. This is a `.json` file with predefined structure.

### The description of the parameters of the algorithm

|         **Parameter**        |                            **Description**                            |                                                                                       **Options/Valid range of values**                                                                                      |
|:----------------------------:|:---------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `initial_graphs`             |                Initial population for genetic algorithm               |`ER` - start from ER-0.5 population                                                                                                                  |
| `mutation_prob`              | Probability of occurrence of mutation in a successive child           | `[0,1]`                                                                                                                                                                                                      |
| `children_at_each_iteration` | Number of children sampled during each attempt                        | Any integer value, default is `30`                                                                                                                                                                           |
| `early_stopping_steps`       | Steps without improvement for early stopping mechanism                | Any integer, default is off                                                                                                                                                                                  |
| `total_attempts`           | Total amount of evolutional steps                                     | Any integer, common values are `1000`, `2000`, `5000`                                                                                                                                                        |
| `max_failed_attempts`           | Maximal number of consequtive failed attempts                                     | Any integer, common values are `250`, `500`                                                                                                                                                       |
