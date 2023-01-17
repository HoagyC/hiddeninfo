import os
import pickle
import sys
from typing import Tuple, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cub_utils import download_from_aws, list_aws_files, upload_to_aws
from cub_classes import TTI_Config, TTI_Output
from tti import run_tti, graph_tti_output, graph_multi_tti_output


def get_most_recent(run_name: str) -> str:
    """Get the most recent run folder for a given run name."""
    run_files = list_aws_files(run_name, get_folders=True)
    return run_files[-1]

def get_all_runs(run_name) -> List[str]:
    """Get all run folders for a given run name."""
    return list_aws_files(run_name, get_folders=True)

def process_run_name(model_file: str) -> Tuple[float, int, str]:
    """Get the sparsity, coef, and last run date from a model file name."""
    split_name = model_file.split("/")
    model_name = split_name[-3]
    model_date = split_name[-2]
    # Get the sparsity and coef
    coef, sparsity = model_name.split("-")
    coef = float(coef.replace("sparsemultimodel", ""))
    sparsity = int(sparsity)
    return coef, sparsity, model_date

def process_results(runs_list: List[str], process_all: bool = False, reprocess: bool = False) -> None:
    """Process TTI results for a list of runs and upload results and graphs to AWS."""
    # Get the most recent run for each run name
    if process_all:
        run_folders = []
        for run in runs_list:
            run_folders.extend(get_all_runs(run))
    else:
        run_folders = [get_most_recent(run) for run in runs_list]


    for folder in run_folders:
        model_file = f"{folder}final_model.pth"

        # Check if results have already been processed
        if not reprocess and folder + 'tti_results.pkl' in list_aws_files(folder, get_folders=False):
            print(f"Results for {folder} already processed")
            continue
    
        download_from_aws([model_file])

        # Get the sparsity and coef from the file name
        model_folder = os.path.join(*model_file.split("/")[:-1])
        coef, sparsity, _ = process_run_name(model_file)

        # Create the TTI config
        tti_config = TTI_Config(
            log_dir=model_folder,
            model_dir=model_file,
            multimodel=True,
            use_attr=True,
            bottleneck=True
        )

        print(f"Running tti with folder {model_folder}, file {model_file}, sparisty {sparsity}, attn_coef {coef}")
        # Run TTI and graph results
        results = run_tti(tti_config)
        graph_tti_output(results, save_dir=model_folder, show=False)

        # Save results and graph
        with open(f"{model_folder}/tti_results.pkl", "wb") as f:
            pickle.dump(results, f)
        upload_files = [f"{model_folder}/tti_results.pkl", f"{model_folder}/tti_results.png"]
        for filename in upload_files:
            upload_to_aws(filename)


def get_results_pkls(runs_list: List[str], use_all: bool = False) -> List[TTI_Output]:
    """Get the TTI results for a list of runs."""
    # Get the most recent run for each run name
    if use_all:
        run_folders = []
        for run in runs_list:
            run_folders.extend(get_all_runs(run))
    else:
        run_folders = [get_most_recent(run) for run in runs_list]
    tti_pkls = [f"{folder}tti_results.pkl" for folder in run_folders]
    download_from_aws(tti_pkls)

    all_results = []
    for pkl_loc in tti_pkls:
        # Get the sparsity and coef
        coef, sparsity, name = process_run_name(pkl_loc)
        with open(pkl_loc, "rb") as f:
            run_results = pickle.load(f)
        
        all_results.append(TTI_Output(coef=coef, sparsity=sparsity, result=run_results, model_name=name))
    
    return all_results




if __name__ == "__main__":
    # List of models to download from AWS (getting the most recent one in each case
    runs_list = [
        "out/sparsemultimodel1-10",
        "out/sparsemultimodel10-10",
    ]
    #     "out/sparsemultimodel0.1-3",
    #     "out/sparsemultimodel1-3",
    # ]

    process_results(runs_list, process_all=True, reprocess=True)
    results = get_results_pkls(runs_list, use_all=True)
    print(len(results))
    graph_multi_tti_output(results)