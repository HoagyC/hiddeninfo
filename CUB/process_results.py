import os
import pickle
import sys
import dataclasses
from typing import Tuple, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.cub_utils import download_from_aws, list_aws_files, upload_to_aws
from CUB.cub_classes import TTI_Config, TTI_Output
from CUB.tti import run_tti, graph_tti_output, graph_multi_tti_output

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
    coef_str, sparsity_str = model_name.split("-")
    coef= float(coef_str.replace("sparsemultimodel", ""))
    sparsity = int(sparsity_str)
    return coef, sparsity, model_date


def create_tti_cfg(model_file: str, model_folder: str) -> TTI_Config:
    """Create the TTI config based on the config.pkl of the original experiment."""

    download_from_aws([model_folder + "/config.pkl"])
    with open(model_folder + "/config.pkl", "rb") as f:
        exp_cfg = pickle.load(f)

    tti_config = TTI_Config(
        log_dir=model_folder,
        model_dir=model_file,
        multimodel=exp_cfg.multimodel,
        data_dir=exp_cfg.data_dir,
        multimodel_type="mixture",
    )

    return tti_config

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
        # model_file = f"{folder}final_model.pth"
        model_file = f"{folder}final_model.pth"

        # Check if results have already been processed
        if not reprocess and folder + 'tti_results_mix.pkl' in list_aws_files(folder, get_folders=False):
            print(f"Results for {folder} already processed")
            continue
    

        # Get the sparsity and coef from the file name
        model_folder = os.path.join(*model_file.split("/")[:-1])
        # coef, sparsity, _ = process_run_name(model_file)
        # print(f"Running tti with folder {model_folder}, file {model_file}, sparisty {sparsity}, attn_coef {coef}")


        # Create the TTI config
        tti_config = create_tti_cfg(model_file, model_folder)
        if not tti_config.multimodel: # careful, it's now saving in results_mix, and is setting tti_conf.multitype to mixture
            continue

        # Download the model
        download_from_aws([model_file])

        tti_config = dataclasses.replace(
            tti_config,
            sigmoid=False,
            model_sigmoid=False,
        )
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
    """
    Takes in a list of run names, takes either the most recent run or all timestamps,
    where TTI has already been run and saved as ./tti_results.pkl, downloads these files from
    AWS, and returns the TTI results as a list.
    """
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
    if len(sys.argv) > 1:
        run_name = sys.argv[1]
        runs_list = ["big_run/" + run_name]
    else:
        # List of models to download from AWS (getting the most recent one in each case
        runs_list = list_aws_files("big_run", get_folders=True)
        print(runs_list)
        runs_list.remove("big_run/postpre_test_sparse/") # still running

    print(f"Processing {runs_list}")
    process_results(runs_list, process_all=False, reprocess=True)
