from datetime import datetime
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.models import Multimodel
from CUB.cub_classes import TTI_Config
from CUB.inference import eval
from CUB.configs import multi_inst_cfg

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
    seq_eval = eval(seq_tti_config)
    print("Joint inst")




def compose_multi():
    j_inst = "big_run/joint_inst/20230224-124648/final_model.pth"
    j_inst_sparse = "big_run/joint_inst_sparse/20230227-161614/final_model.pth"

    # Make multimodel from the two joint models
    j_inst_model = torch.load(j_inst)
    j_inst_sparse_model = torch.load(j_inst_sparse)


    multimodel = Multimodel(multi_inst_cfg)
    multimodel.pre_models = nn.ModuleList([j_inst_model.first_model, j_inst_sparse_model.first_model])
    multimodel.post_models = nn.ModuleList([j_inst_model.second_model, j_inst_sparse_model.second_model])

    DATETIME_FMT = "%Y%m%d-%H%M%S"
    now_str = datetime.now().strftime(DATETIME_FMT)

    save_dir = "big_run/multi_inst_joint/" + now_str
    os.makedirs(save_dir, exist_ok=True)

    torch.save(multimodel, os.path.join(save_dir, "final_model.pth"))

if __name__ == "__main__":
    get_attrs()