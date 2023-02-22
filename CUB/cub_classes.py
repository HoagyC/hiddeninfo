import dataclasses
from typing import Optional, List, Tuple

from CUB.config import N_ATTRIBUTES, N_CLASSES


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@dataclasses.dataclass
class RunRecord:
    epoch: int = -1
    loss: float = float("inf")
    acc: float = 0.0


@dataclasses.dataclass
class Meters:
    loss: AverageMeter = dataclasses.field(default_factory=AverageMeter)
    label_acc: AverageMeter = dataclasses.field(default_factory=AverageMeter)
    concept_acc: AverageMeter = dataclasses.field(default_factory=AverageMeter)


@dataclasses.dataclass
class Experiment:
    tag: str = "basic"
    exp: str = "multimodel"
    seed: int = 1

    # Data
    log_dir: str = "big_out"
    data_dir: str = "CUB_masked_class"
    image_dir: str = "images"
    save_step: int = 100

    # Model
    multimodel: bool = False
    n_attributes: int = 109
    num_classes: int = N_CLASSES
    expand_dim: int = 500
    use_relu: bool = False
    use_sigmoid: bool = False
    pretrained: bool = True
    use_aux: bool = True
    post_model_dropout: Optional[float] = None

    # Training
    epochs: int = 1000
    optimizer: str = "SGD"
    scheduler_step: int = 1000
    attr_loss_weight: float = 1.0
    lr: float = 1e-03
    weight_decay: float = 2e-4
    attr_sparsity: int = 1
    batch_size: int = 64

    tti_int: int = 10 # Frequency of running TTI during run

    freeze: bool = False
    weighted_loss: str = ""
    uncertain_labels: bool = False

    # Multi-model specific parameters
    shuffle_models: bool = False
    freeze_post_models: bool = False
    freeze_pre_models: bool = False
    reset_pre_models: bool = False
    reset_post_models: bool = False
    n_models: int = 4
    thin: bool = False # Whether to only change the last/first layers of th e


    quick: bool = False

    model_sigmoid: bool = True
    gen_pred_sigmoid: bool = False
    use_test: bool = False # Whether to also check test accuracy as we go

    n_alternating: int = 1 # How many times to alternate between pre and post models
    freeze_first: str = "pre" # Whether to freeze the first pre or post model, "pre" or "post"
    alternate_reset: bool = True 


@dataclasses.dataclass
class TTI_Config:
    log_dir: str = "."  # where results are stored
    model_dir: str = ""

    eval_data: str = "test"  # whether to use test or val data
    batch_size: int = 16

    use_sigmoid: bool = False
    attr_sparsity: int = 1

    data_dir: str = "CUB_masked_class"  # directory to the data used for evaluation
    data_dir_raw: str = "CUB_processed"  # directory to the raw data
    n_attributes: int = 109
    image_dir: str = "images"  # test image folder to run inference on

    # whether to print out performance of individual atttributes
    feature_group_results: bool = False
    # Whether to correct with class or instance level values
    replace_class: bool = True

    n_trials: int = 1  # Number of trials to run, when mode is random
    n_groups: int = 28  # n. groups of attributes (28 enums with ~10 options = 312 attrs)
    multimodel: bool = True # whether to use the multimodel architecture

    sigmoid: bool = False
    model_sigmoid: bool = False

    flat_intervene: bool = True
    intervene_vals: Tuple[float, float] = (-3, 3) # If intervening with a flat value, what is it?

base_ind_tti_cfg = TTI_Config(
    n_trials=5,
    replace_class=True,
    use_sigmoid=True,
    log_dir="TTI_ind",
)

@dataclasses.dataclass
class TTI_Output:
    coef: float
    sparsity: int
    result: List[Tuple[int, float]]
    model_name: Optional[str] = None

ind_cfg = Experiment(
    tag="ind_XtoC",
    exp="Independent",
    epochs=1000,
    lr=0.01,
    weighted_loss="multiple"
)


seq_cfg = Experiment(
    tag="seq_CtoY",
    exp="Sequential",
    epochs=1000,
    lr=0.001,
    weighted_loss="multiple",
)

joint_cfg = Experiment(
    tag="joint",
    exp="Joint",
    epochs=1000,
    weighted_loss="multiple",
    attr_loss_weight=0.01,
    lr=0.001,
)

multiple_cfg1 = Experiment(
    multimodel=True,
    tag='multimodel0.01',
    exp="Multimodel",
    epochs=150,
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

prepost_cfg = dataclasses.replace(
    multiple_cfg3,
    exp="Alternating",
    tag="prepost_test",
    n_alternating=2,
    freeze_first="pre",
    epochs=50,
    tti_int=10,
    data_dir="CUB_instance_masked",
)

postpre_cfg = dataclasses.replace(
    multiple_cfg3,
    exp="Alternating",
    tag="postpre_test",
    n_alternating=2,
    freeze_first="post",
    epochs=50,
    tti_int=10,
    data_dir="CUB_instance_masked",
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
    prepost_cfg, # Retrains pre model, then post model, twice
    postpre_cfg, # Retrains post model, then pre model, twice
]

sparsities = [1, 3, 10]
n_runs = 3

# Creating a list of configs for each sparsity
configs = []
for sparsity in sparsities:
    for cfg in raw_configs:
        configs.append(dataclasses.replace(cfg, attr_sparsity=sparsity))



