import argparse
import os
import logging
import sys
import pandas as pd

from pathlib import Path

from generation import get_initial_graphs

from greedy import generate_max_volume_greedy
from genetics import EvolutionalAlgorithm
from utils import Config, save_pickle, read_pickle, time_wrapper, save_json

from typing import Union

import tempfile

BASIC_STRING = "\n\n\n\t\t\t{}\n\n\n" + '='*150


def get_parser() -> argparse.ArgumentParser :
    root = argparse.ArgumentParser()
    
    root.add_argument("--config", type=Path, help="Config in .json format", required=True)
    root.add_argument("-o", "--outdir", type=Path, help="Output directory")
    root.add_argument("-t", "--threads", type=int, help="Threads (10)", default=min(10, os.cpu_count()))
    root.add_argument("--mode", choices=["greedy", "random", "all", "genetic", "localopt"], 
                      default="all", 
                      help="Option of generation graphs")
    root.add_argument("--analyze", help="output basic analysis", action="store_true")
    
    root.add_argument("--force", action="store_true", help="Recreate entire dir")
    
    root.add_argument("--volume", choices=["avg", "min", "energy"], required=True, type=str, help="Function for computing volume")
    root.add_argument("-g", "--graphs", type=Path, required=False, 
                      help="User-defined list of graphs in .pkl format. It can be initial input for greedy algorithm, genetic algorithm, local optimization algorithm")
    
    root.add_argument("--save_generated", action="store_true", default=False, help="Whether to save initial set of graphs")
    
    root.add_argument("--log", 
                      type=Path, 
                      default=None, 
                      help="""Path to the log file, which will contain logs from console, if this argument is provided. 
                                If the argument isn't provided, log file will be generated automatically in the 'outdir' directory with the name 'logs.log'""")
    
    return root

def get_logger(outdir: Path, logfile: Union[Path, str, None]="log.log") -> logging.Logger:
    
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    
    
    stringfmt = "[%(asctime)s.%(msecs)03d] [%(threadName)s] [%(name)s] [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=stringfmt, datefmt=datefmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    
    if logfile:
        file_output = logfile
    else:
        file_output = outdir / "log.log"
    
    
    file_handler = logging.FileHandler(file_output, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)



    root.addHandler(console_handler)
    root.addHandler(file_handler)
    
    return root



@time_wrapper
def main():
    parser: argparse.ArgumentParser = get_parser()
    
    args = parser.parse_args()
    save_root = Path(args.outdir)
    save_root.mkdir(parents=True, exist_ok=args.force)
    
    root: logging.Logger = get_logger(outdir=args.outdir, logfile=args.log)
    
    
    root.info("Initialized logger")
    
    root.debug(f"The arguments from the console command are: {args}")

    
    config = Config(config_file=args.config)
    root.debug(f"Read arguments from config file from {args.config}. Config will be duplicated after script finishes.")    
    

    
    distances_set = set(config.distances_configurations.keys())
    
    graphs = None
    
    if args.graphs:
        graphs = read_pickle(filename=args.graphs)
    
    greedy_graphs_per_distance = None
    
    tmpdir = None
    if 8 * config.nodes_number ** 2 * config.greedy_sampling_size > 5e10:
        root.info("Estimated number of bytes for storing adjacencies in the RAM is too high (> 50Gb), will use a temporary directory for it!")
        
        tmpdir = tempfile.mkdtemp()
        root.info(f"Temporary directory: {tmpdir}")
        
        
        
        
    if args.mode in {"all", "greedy"} or any(((args.mode in {"genetic"} and "greedy" in config.genetic_config["initial_graphs"]),
                                             (args.mode in {"localopt"} and "greedy" in config.localopt_config["initial_graphs"])
                                             )):
        
        initial_graphs_greedy = get_initial_graphs(
            config=dict(initial_graphs=config.greedy_initial_graphs, 
                        models=config.greedy_models, 
                        greedy_sampling_size=config.greedy_sampling_size),
            threads=args.threads,
            distances_set=distances_set,
            greedy_graphs_objects_per_distance=greedy_graphs_per_distance,
            samples=config.samples,
            nodes_number=config.nodes_number,
            orca_path=config.orca_path,
            equal_sizes=config.equal_sizes,
            maybe_ready_graphs=graphs,
            tmpdir=tmpdir
            )
        
        if args.save_generated:
            save_pickle(obj=initial_graphs_greedy, to_filename=save_root/"initial_graph_objects_per_model.pkl")
            root.info("Saved initial set of graphs for greedy")
        
        root.info(BASIC_STRING.format("Started greedy generation"))
        
        _, greedy_graphs_per_distance = generate_max_volume_greedy(
            initial_graphs=initial_graphs_greedy,
            number_of_graphs=config.samples,
            n_jobs=args.threads,
            distances_configs=config.greedy_distances,
            volume_func_type=args.volume,
            save_dir=args.outdir,
        )

    if args.mode == "genetic":
        root.info(BASIC_STRING.format("Started genetic generation"))

        initial_graphs = get_initial_graphs(
            config=config.genetic_config | {"models" : config.models_configurations, "greedy_sampling_size": config.samples},
            threads=args.threads,
            distances_set=distances_set,
            greedy_graphs_objects_per_distance=greedy_graphs_per_distance,
            samples=config.samples,
            nodes_number=config.nodes_number,
            orca_path=config.orca_path,
            equal_sizes=config.equal_sizes,
            maybe_ready_graphs=graphs,
            tmpdir=tmpdir

        )            
        
        evo_algo = EvolutionalAlgorithm(
            number_of_nodes=config.nodes_number,
            initial_graphs=initial_graphs,
            genetic_config=config.genetic_config,
            distances_configurations=config.distances_configurations,
            n_jobs=args.threads,
            save_dir=save_root,
            orca_path=config.orca_path,
            volume_func_type=args.volume,
            checkpoint_steps_interval=config.genetic_checkpoint_interval,
        )
        
        evo_algo.run_every_combination_of_evolutional_algo()

        save_pickle(obj=evo_algo.metareports, to_filename=save_root / "genetic_metareports.pkl")

        evo_algo.global_report.to_csv(save_root / "genetic_runs_finished.csv", index=False)
    
    if args.mode == "localopt":
        from localopt import optimize_set_of_graphs_locally
        
        initial_graphs = get_initial_graphs(
            config=config.localopt_config,
            threads=args.threads,
            distances_set=distances_set,
            greedy_graphs_objects_per_distance=greedy_graphs_per_distance,
            samples=config.samples,
            nodes_number=config.nodes_number,
            orca_path=config.orca_path,
            equal_sizes=config.equal_sizes,
            maybe_ready_graphs=graphs,
            tmpdir=tmpdir

        )
        report: pd.DataFrame = optimize_set_of_graphs_locally(config=config.localopt_config,
                                                number_of_nodes=config.nodes_number,
                                                number_of_graphs=config.samples,
                                                distances_configurations=config.distances_configurations,
                                                n_jobs=args.threads,
                                                save_dir=args.outdir,
                                                orca_path=config.orca_path,
                                                initial_graphs=initial_graphs,
                                                volume_func_type=args.volume,
                                                )
        
        report.to_csv(args.outdir / "report.csv")
                
    save_json(obj=config.settings, filename=save_root / "config.json")
    root.info(f"Config and outputs were saved at {save_root}")
    
    
    if tmpdir is not None:
        import shutil
        shutil.rmtree(tmpdir)

        root.info("Temporary directory has been cleaned up.")

    
if __name__ == '__main__':
    main()