from datetime import datetime
import os
import shutil
import sys
from typing import List

from matplotlib import pyplot as plt
import pandas as pd

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.models import Multimodel, CUB_Multimodel
from CUB.cub_classes import TTI_Config
from CUB.inference import run_eval
from CUB.configs import multi_inst_cfg
from CUB.dataset import load_data
from CUB.cub_utils import download_from_aws, download_folder_from_aws

def get_attrs():
    seq_inst_path = "big_run/seq_inst/20230224-140619/final_model.pth"
    joint_inst_path = "big_run/joint_inst/20230224-124648/final_model.pth"
    joint_inst_sparse_path = "big_run/joint_inst_sparse/20230227-161614/final_model.pth"

    seq_inst_model = torch.load(seq_inst_path)
    joint_inst_model = torch.load(joint_inst_path)
    joint_inst_sparse_model = torch.load(joint_inst_sparse_path)

    seq_tti_config = TTI_Config(
        log_dir=seq_inst_path.split("/")[:-1],
        model_dir=seq_inst_path,
        multimodel=False,
        data_dir="CUB_instance_masked",
    )

    print("Seq inst")
    seq_eval =run_eval(seq_tti_config)
    print("Joint inst")


def compose_multi(models_list): # List of paths to models
    # Make multimodel from the two joint models
    for path in models_list:
         # Check if each path exists, if not, try to download from aws
        if not os.path.exists(path):
            has_downloaded = download_from_aws([path])
            if not has_downloaded:
                raise FileNotFoundError(f"Could not find {path} and could not download from aws")
    
    sep_models = [torch.load(model_path) for model_path in models_list]

    multimodel = Multimodel(multi_inst_cfg)
    breakpoint()
    multimodel.pre_models = nn.ModuleList([model.first_model for model in sep_models])
    multimodel.post_models = nn.ModuleList([model.second_model for model in sep_models])

    DATETIME_FMT = "%Y%m%d-%H%M%S"
    now_str = datetime.now().strftime(DATETIME_FMT)

    save_dir = "out/multi_attr_weight/" + now_str
    os.makedirs(save_dir, exist_ok=True)

    torch.save(multimodel, os.path.join(save_dir, "final_model.pth"))


def look_at_activations():
    model_with_two_joints_path = "big_run/multi_inst_joint/20230302-141428/final_model.pth"
    seq_model_path = "big_run/seq_inst/20230224-124623/final_model.pth"
    seq_sparse_model_path = "big_run/seq_inst_sparse/20230227-183548/final_model.pth"
    multi_path = "big_run/multimodel_inst/20230224-172742/final_model.pth"

    train_data_path = "CUB_instance_masked/train.pkl"
    val_data_path = "CUB_instance_masked/val.pkl"
    test_data_path = "CUB_instance_masked/test.pkl" 

    args = multi_inst_cfg

    # Looking at each model's activations
    train_loader = load_data([train_data_path], args)
    val_loader = load_data([val_data_path], args)
    test_loader = load_data([test_data_path], args)

    seq_attrs = []
    seq_sparse_attrs = []
    joint_attrs = []
    multi_attrs = []

    model_paths = [seq_model_path, seq_sparse_model_path, model_with_two_joints_path, multi_path]
    output_lists = [seq_attrs, seq_sparse_attrs, joint_attrs, multi_attrs]
    with torch.no_grad():
        
            for model_path, olist in zip(model_paths, output_lists):
                model = torch.load(model_path)
                model.eval()
                for batch in train_loader:
                    inputs, class_labels, attr_labels, attr_mask = batch

                    attr_labels = [i.float() for i in attr_labels]
                    attr_labels = torch.stack(attr_labels, dim=1)

                    attr_labels = attr_labels.cuda() if torch.cuda.is_available() else attr_labels
                    inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                    class_labels = class_labels.cuda() if torch.cuda.is_available() else class_labels
                    attr_mask = attr_mask.cuda() if torch.cuda.is_available() else attr_mask

                    output = model.generate_predictions(inputs, attr_labels, attr_mask)
                    olist.append(output[0])

            del(model) # Clear memory
            print(f"Done with model {model_path.split('/')[-3]}")


    seq_attrs = torch.cat(seq_attrs, dim=1)
    seq_sparse_attrs = torch.cat(seq_sparse_attrs, dim=1)
    joint_attrs = torch.cat(joint_attrs, dim=1)
    multi_attrs = torch.cat(multi_attrs, dim=1)

    # Looking at the difference between the two joint models
    seq_attrs = seq_attrs[0]
    seq_sparse_attrs = seq_sparse_attrs[0]
    joint_attrs1 = joint_attrs[0]
    joint_attrs2 = joint_attrs[1]
    multi_attrs1 = multi_attrs[0]
    multi_attrs2 = multi_attrs[1]

    diff1 = joint_attrs1 - seq_attrs
    diff2 = joint_attrs2 - seq_attrs

    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    print("Cosine similarity between joint1 and seq")
    print(cos_sim(joint_attrs1, seq_attrs).mean())
    print("Cosine similarity between joint2 and seq")
    print(cos_sim(joint_attrs2, seq_attrs).mean())
    print("Cosime similiary of the deviations from seq")
    print(cos_sim(diff1, diff2).mean())
    print("Cosine similarity of seq and seq_sparse")
    print(cos_sim(seq_attrs, seq_sparse_attrs).mean())


    breakpoint()

def concat_wandb_runs():
    # Concatenate the runs from wandb after training in separate stages

    import wandb
    api = wandb.Api()
    project_str = "/distillrepr/distill_CUB/runs/"
    run_ids = ["k1iluazx", "3qwc211z"]
    paths = [project_str + run_id for run_id in run_ids]

    # Get the run histories
    dfs = []
    for path in paths:
        run = api.run(path)
        data = run.history() # Returns a pandas dataframe
        dfs.append(data)

    # Concatenate them so that the epochs of the second run are after the first, so that the second run starts from epoch (last epoch of first run + 1)

    # Add the total epochs of the first run to the second run's epochs
    dfs[1]["epoch"] = dfs[1]["epoch"] + dfs[0]["epoch"].max() + 1
    
    # Concatenate the two dataframes
    df = pd.concat(dfs)
    

    # Graph using matplotlib with the x axis being the epoch and the y axis having "train_acc", "val_acc", "train_cross_acc0"

    # First check that the epochs are in order
    assert df["epoch"].is_monotonic_increasing
    # then check that the columns exist as expected
    assert "train_acc" in df.columns, f"train_acc not in {df.columns}"
    assert "val_acc" in df.columns, f"val_acc not in {df.columns}"
    assert "train_cross_acc0" in df.columns, f"train_cross_acc0 not in {df.columns}"

    # train_cross_acc0 is written on different lines to train_acc, so we need to collapse so there's one line for each value of epoch, making sure to ignore NaNs
    df = df.groupby("epoch").mean()

    # Plot the graph, with the x axis being the epoch and the y axis having "train_acc", "val_acc", "train_cross_acc0"
    # Note that epoch is the index of the dataframe, so we don't need to specify it
    df[150:].plot(y=["train_acc", "val_acc", "train_cross_acc0", "val_cross_acc0"])
    print(df["train_cross_acc0"])
    
    # Save figure
    plt.savefig("images/wandb_concat.png")


def save_all_premodels(runs_s3_loc: List[str]):
    '''
    Downloading the various premodels that are saved in AWS S3,
    checking their attr_loss_weight and whether they have use_dropout=True,
    then saving them all in a clearly labelled folder in AWS S3
    '''
    new_save_folder = "attr_dropout_base"
    os.makedirs(new_save_folder, exist_ok=True)

    for run in runs_s3_loc:
        download_folder_from_aws(run) # Downloads to the same folder as in s3
        model_path = run + "latest_model.pth"
        config_pkl = run + "config.pkl"
        model: CUB_Multimodel
        model = torch.load(model_path)
        attr_loss_weight = model.args.attr_loss_weight
        uses_dropout = model.args.use_pre_dropout

        if not isinstance(attr_loss_weight, list):
            attr_loss_weight = [attr_loss_weight] * 2 # Models are all n_model=2 but earlier models only had one attr_loss_weight
        
        # Save the premodel
        for ndx, premodel in enumerate(model.pre_models):
            weight_str = str(attr_loss_weight[ndx]).replace(".", "_")
            if model.args.exp == "MultiSequential":
                weight_str = "sep"
            premodel_folder = f"{new_save_folder}/premodel_attr_loss_weight_{weight_str}_drop_{uses_dropout}"
            if os.path.exists(premodel_folder):
                print(f"File {premodel_folder} already exists, skipping")
                continue

            os.makedirs(premodel_folder, exist_ok=True)
            config_path = os.path.join(premodel_folder, "config.pkl")
            shutil.copyfile(config_pkl, config_path)

            premodel_path = os.path.join(premodel_folder, "premodel.pth")
            torch.save(premodel, premodel_path)
    



if __name__ == "__main__":
    run_s3_locs = [
        "out/multi_attr_loss_weight_0.1,10_drop/20230324-183105/",
        "out/multi_attr_loss_weight_0.1,10/20230324-183145/",
        "out/multi_attr_loss_weight_0.2,5_drop/20230324-182851/",
        "out/multi_attr_loss_weight_0.2,5/20230324-183119/",
        "out/multi_attr_loss_weight_0.3,3_drop/20230324-180429/",
        "out/multi_attr_loss_weight_0.5,2_drop/20230324-182823/",
        "out/multimodel_post_inst/20230323-165608/",
        "out/multimodel_post_inst/20230323-165611/",
        "out/multimodel_seq/20230320-185823/", # seq 1 and 1 drop
        "out/multimodel_post_inst/20230320-124841/", # multi 1 and 1 drop
        "out/multimodel_post_inst/20230320-130430/", # multi 1 and 1 no drop
    ]

    save_all_premodels(run_s3_locs)


### Old scripts graveyard
# timestamps = [
#     "20230322-174145/",
#     "20230322-183638/",
# ]

# folder = "out/multimodel_post_inst/"
# paths = [folder + timestamp + "latest_model.pth" for timestamp in timestamps]

# compose_multi(paths)

# # Making multiple pairs of multimodels
# joint_2paths1 = [folder + timestamp + "latest_model.pth" for timestamp in joint_timestamps[:2]]
# joint_2paths2 = [folder + timestamp + "latest_model.pth" for timestamp in joint_timestamps[2:]]
# compose_multi(joint_2paths1)


# paths = [
#     "out/multiseq_p1.csv",
#     "out/multiseq_p2.csv",
# ]
# concat_wandb_runs()







