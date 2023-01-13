import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cub_utils import download_from_aws, list_files, upload_to_aws
from cub_classes import TTI_Config
from tti import run_tti, graph_tti_output

def process_results() -> None:
    # List of models to download from AWS (getting the most recent one in each case)

    runs_list = [
      "out/sparsemultimodel0.1-10",
      "out/sparsemultimodel1-10",
      "out/sparsemultimodel10-10"
    ]

    last_run_folder = [list_files(run)[-1] for run in runs_list]
    model_files = [f"{folder}final_model.pth" for folder in last_run_folder]
    download_from_aws(model_files)

    for model_file in model_files:
        # Get the model name
        model_folder = os.path.join(*model_file.split("/")[:-1])
        model_name = model_file.split("/")[-3]
        # Get the sparsity and coef
        coef, sparsity = model_name.split("-")
        coef = float(coef.replace("sparsemultimodel", ""))
        sparsity = int(sparsity)

        # Create the TTI config
        tti_config = TTI_Config(
            log_dir=model_folder,
            model_dir=model_file,
            multimodel=True,
            use_attr=True,
            bottleneck=True
        )

        print(f"Running tti with folder {model_folder}, file {model_file}, sparisty {sparsity}, attn_coef {coef}")

        results = run_tti(tti_config)
        with open(f"{model_folder}/tti_results.pkl", "wb") as f:
            pickle.dump(results, f)
        graph_tti_output(results, save_dir=model_folder, show=False)
        upload_to_aws([f"{model_folder}/tti_results.pkl", f"{model_folder}/tti_graph.png"])



if __name__ == "__main__":
    process_results()