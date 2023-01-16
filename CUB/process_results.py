import os
import pickle
import sys
from typing import Tuple, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cub_utils import download_from_aws, list_files, upload_to_aws
from cub_classes import TTI_Config, TTI_Output
from tti import run_tti, graph_tti_output, graph_multi_tti_output


def get_most_recent(run_name):
    """Get the most recent run folder for a given run name."""
    run_files = list_files(run_name)
    return run_files[-1]

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

def process_results(runs_list: List[str]) -> None:
    """Process TTI results for a list of runs and upload results and graphs to AWS."""
    # Get the most recent run for each run name
    last_run_folder = [get_most_recent(run) for run in runs_list]
    model_files = [f"{folder}final_model.pth" for folder in last_run_folder]
    download_from_aws(model_files)

    for model_file in model_files:
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


def get_results_pkls(runs_list: List[str]) -> List[TTI_Output]:
    """Get the TTI results for a list of runs."""
    # Get the most recent run for each run name
    last_run_folder = [get_most_recent(run) for run in runs_list]
    tti_pkls = [f"{folder}tti_results.pkl" for folder in last_run_folder]
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
        "out/sparsemultimodel0.1-10",
        "out/sparsemultimodel1-10",
        "out/sparsemultimodel10-10",
    ]
    #     "out/sparsemultimodel0.1-3",
    #     "out/sparsemultimodel1-3",
    # ]


    results = get_results_pkls(runs_list)
    graph_multi_tti_output(results)