import dataclasses
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.cub_classes import Experiment

ind_cfg = Experiment(
    tag="ind_XtoC",
    exp="Independent",
    epochs=[1000, 1000],
    lr=0.01,
    weighted_loss="multiple"
)


seq_cfg = Experiment(
    tag="seq_CtoY",
    exp="Sequential",
    epochs=[1000, 1000],
    lr=0.001,
    weighted_loss="multiple",
)

joint_cfg = Experiment(
    tag="joint",
    exp="Joint",
    epochs=[1000],
    weighted_loss="multiple",
    attr_loss_weight=0.01,
    lr=0.001,
)

multiple_cfg1 = Experiment(
    multimodel=True,
    tag='multimodel0.01',
    exp="Multimodel",
    epochs=[150, 150],
    weighted_loss="multiple",
    lr=0.001,
    batch_size=24,
    attr_loss_weight=0.01,
    # multimodel specific
    n_models=2,
    freeze_post_models=True,
    reset_pre_models=True,
)

multiple_cfg2 = dataclasses.replace(
    multiple_cfg1,
    tag='multimodel0.1',
    attr_loss_weight=0.1,
)

multiple_cfg3 = dataclasses.replace(
    multiple_cfg1,
    tag='multimodel1.0',
    attr_loss_weight=1.0,
)

ind_sparse_cfg = dataclasses.replace(
    ind_cfg,
    tag="ind_sparse",
    attr_sparsity=10,
    batch_size=64 * 10,
)

seq_sparse_cfg = dataclasses.replace(
    seq_cfg,
    tag="seq_sparse",
    attr_sparsity=10,
    batch_size=64 * 10,
)

joint_sparse_cfg = dataclasses.replace(
    joint_cfg,
    tag="joint_sparse",
    attr_sparsity=10,
    attr_loss_weight=1.0,
)

joint_cfg2 = dataclasses.replace(
    joint_cfg,
    tag="joint_0.1",
    attr_loss_weight=0.1,
)

joint_cfg3 = dataclasses.replace(
    joint_cfg,
    tag="joint_1.0",
    attr_loss_weight=1.0,
)

multi_sparse_cfg = dataclasses.replace(
    multiple_cfg3,
    tag="multimodel_sparse",
    attr_sparsity=10,
    attr_loss_weight=1.0,
)

thin_cfg = dataclasses.replace(
    multiple_cfg3,
    tag = "thin",
    thin=True
)

# Creating normal runs but using CUB_instance_masked instead of CUB_masked_class as the args.dir
ind_inst_cfg = dataclasses.replace(
    ind_cfg,
    tag="ind_inst",
    data_dir="CUB_instance_masked",
)

seq_inst_cfg = dataclasses.replace(
    seq_cfg,
    tag="seq_inst",
    data_dir="CUB_instance_masked",
)

joint_inst_cfg = dataclasses.replace(
    joint_cfg3,
    tag="joint_inst",
    data_dir="CUB_instance_masked",
)

multi_inst_cfg = dataclasses.replace(
    multiple_cfg3,
    tag="multimodel_inst",
    data_dir="CUB_instance_masked",
)

multi_inst_post_cfg = dataclasses.replace(
    multiple_cfg3,
    tag="multimodel_post_inst",
    data_dir="CUB_instance_masked",
    freeze_post_models=False,
    reset_pre_models=False,
    freeze_pre_models=True,
    reset_post_models=True,
)

multi_noreset_cfg = dataclasses.replace(
    multi_inst_cfg,
    tag="multimodel_noreset",
    reset_pre_models=False,
)

multi_noreset_post_cfg = dataclasses.replace(
    multi_inst_post_cfg,
    tag="multimodel_noreset_post",
    reset_post_models=False,
)

all_frozen = dataclasses.replace(
    multiple_cfg3,
    tag="all_frozen",
    freeze_post_models=True,
)

multi_seq_cfg = dataclasses.replace(
    multi_inst_cfg,
    tag="multimodel_seq",
    exp="MultiSequential",
)

prepost_cfg = dataclasses.replace(
    multi_inst_cfg,
    exp="Alternating",
    tag="prepost_test",
    n_alternating=2,
    freeze_first="pre",
    epochs=[150, 20, 20, 10, 10],
    data_dir="CUB_instance_masked",
    alternating_reset=False,
    do_sep_train=True,
)

postpre_cfg = dataclasses.replace(
    multi_inst_cfg,
    exp="Alternating",
    tag="postpre_test",
    n_alternating=2,
    freeze_first="post",
    epochs=[150, 20, 20, 10, 10],
    data_dir="CUB_instance_masked",
    alternating_reset=False,
    do_sep_train=True,
)

seq_sparse_class_cfg = dataclasses.replace(
    seq_cfg,
    tag="seq_sparse_class",
    class_sparsity=2,
)

seq_inst_sparse_class_cfg = dataclasses.replace(
    seq_inst_cfg,
    tag="seq_inst_sparse_class",
    class_sparsity=2,
)

joint_missing_attrs_cfg = dataclasses.replace(
    joint_inst_cfg,
    tag="joint_missing_attrs",
    n_attributes=94, # Leaving 4 attr groups out
)

just_xtoc_cfg = dataclasses.replace(
    seq_inst_cfg,
    tag="just_xtoc",
    exp="JustXtoC"
)

multi_seq_tti_check_cfg = dataclasses.replace(
    multi_seq_cfg,
    report_cross_accuracies = True,
    do_sep_train = False,
    tti_int = 10,
    load = "out/multimodel_seq/20230320-185823/final_model.pth",
    use_pre_dropout = False,
)

# Making a new list of configs to test lots of different things
raw_configs = [
    joint_inst_cfg, 
    ind_inst_cfg, 
    seq_inst_cfg, 
    multi_inst_cfg, # Retrains pre model, freezes post model
    multi_inst_post_cfg, # Retrains post model, freezes pre model
    multi_noreset_cfg, # Retrains pre models, without resetting
    multi_noreset_post_cfg, # Retrains post models, without resetting
    prepost_cfg, # Retrains pre model, then post model, twice, no reset
    postpre_cfg, # Retrains post model, then pre model, twice, no reset
]

sparsities = [1, 3, 10]
n_runs = 3

# Creating a list of configs for each sparsity
configs = []
for sparsity in sparsities:
    for cfg in raw_configs:
        configs.append(dataclasses.replace(cfg, attr_sparsity=sparsity))



