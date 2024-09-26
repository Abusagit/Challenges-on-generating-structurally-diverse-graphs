import argparse
from pathlib import Path

from subprocess import Popen, PIPE
from shutil import copyfile
import yaml


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outdir", type=Path, help="output dir")
    parser.add_argument("-g", "--graphs", type=Path, help="input graphs")
    parser.add_argument("-n", "--num_iterations", type=int, help="Number of DiGress iterations")
    parser.add_argument("-d", "--graph_distance", type=str, choices=["GCD", "Portrait", "netLSD_heat", "netLSD_wave"], 
                        default="GCD", help="Graph distance for usage")
    
    parser.add_argument("--memory_transfer", type=str or None, choices=[None, "prev"], default=None,
                        help="""Strategy for preserving model memory between iterations: 
                                `None` - no memory preserved, fresh start, 
                                `prev` - use previous last checkpoint""")
    
    parser.add_argument("--root", type=Path, help="GiGress root directory", default=Path("./"))
    parser.add_argument("-b", "--batch_size", type=int, default=2500)
    parser.add_argument("--graphs_limit", type=int, default=1_000_000, help="Overall limit of graphs to be generated")
    parser.add_argument("--greedy_size", type=int, default=1000)

    
    return parser

def _launch_process(full_args: list[str]):
    proc = Popen(full_args, bufsize=0, stdin=None,
                 stderr=PIPE, universal_newlines=True)
    err_msg = proc.stderr.read().strip()
    proc.stderr.close()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(err_msg)

def launch_DiGress():
    full_args = ["python", "src/main.py"]
    _launch_process(full_args)
    
def launch_greedy_generation(weights_file:Path, number_to_generate:int, 
                             greedy_size:int, graph_distance:str, outdir:Path, 
                             batch_size:int,
                             ):
    
    full_args = ["python", "generate_greedy_from_graph_model.py"] + \
                ["--weights", str(weights_file)] + \
                ["--generate", str(number_to_generate)] + \
                ["--greedy_size", str(greedy_size)] + \
                ["--graph_distance", str(graph_distance)] + \
                ["--outdir", str(outdir)] + \
                ["-b", str(batch_size)]
                
    _launch_process(full_args)


def rewrite_data_config_file(data_config_path, new_datadir, new_graphs_file, prev_run_weights:None or Path=None):
    initial_data_config = yaml.safe_load(open(data_config_path))
    initial_data_config["datadir"] = new_datadir
    initial_data_config["graphs_file"] = str(new_graphs_file)
    
    if prev_run_weights is not None:
        initial_data_config["start_weights"] = prev_run_weights
        
    yaml.dump(initial_data_config, open(data_config_path, "w"))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    # preserve previous state
    tmp_data_config_filename = ".user.yaml"
    data_confg_file = str(args.root / "configs" / "dataset" / "user.yaml")
    copyfile(src=data_confg_file, dst=tmp_data_config_filename)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    
    data_dir_template = "data/user_" + str(args.outdir.name) + "_{}"
    
    
    # get config directory
    _config = yaml.safe_load(open(args.root / "configs" / "config.yaml"))
    checkpoint_dir = _config["hydra"]["run"]["dir"]
    del _config
    

    checkpoint_name_template = "last{}.ckpt"
    digress_output_subdir_template = "digress_run_{}"
    
    
    graphs_per_iteration = args.graphs_limit // args.num_iterations
    
    step_graphs_file = args.graphs
    step_graphs_datadir_for_digress = data_dir_template.format(0)
    prev_run_weights = None
    
    for step in range(args.num_iterations):
        rewrite_data_config_file(data_config_path=data_confg_file, 
                                new_datadir=step_graphs_datadir_for_digress, 
                                new_graphs_file=step_graphs_file,
                                prev_run_weights=prev_run_weights)
        
        launch_DiGress()
        
        checkpoint_name = checkpoint_name_template.format(f"-v{step+1}") #if step > 0 else "last.ckpt"
        digress_checkpoints_path = next((Path(checkpoint_dir) / "checkpoints").iterdir())
        
        digress_output_dir = digress_output_subdir_template.format(step)
        

        
        step_outdir: Path = outdir / digress_output_dir
        step_outdir.mkdir(exist_ok=True, parents=True)
        
        current_step_model_weights = digress_checkpoints_path / checkpoint_name
        launch_greedy_generation(weights_file=current_step_model_weights,
                                 number_to_generate=graphs_per_iteration,
                                 greedy_size=args.greedy_size,
                                 graph_distance=args.graph_distance,
                                 outdir=step_outdir,
                                 batch_size=args.batch_size,
                                 )
        copyfile(data_confg_file, step_outdir / "data_config.yaml")

        step_graphs_file = (step_outdir / "final_graphs_greedy.npy").resolve()
        step_graphs_datadir_for_digress = data_dir_template.format(step + 1)
        
        if args.memory_transfer == "prev":
            prev_run_weights = str(current_step_model_weights.resolve())
    
    
    copyfile(tmp_data_config_filename, data_confg_file) # return it back