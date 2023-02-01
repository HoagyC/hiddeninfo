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
    log_dir: str = "out"
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

    # Training
    epochs: int = 1000
    optimizer: str = "SGD"
    scheduler_step: int = 1000
    attr_loss_weight: float = 1.0
    lr: float = 1e-03
    weight_decay: float = 2e-4
    attr_sparsity: int = 1

    batch_size: int = 64

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

    quick: bool = False


@dataclasses.dataclass
class TTI_Config:
    log_dir: str = "."  # where results are stored
    # where the trained model is saved
    model_dirs: List[str] = dataclasses.field(default_factory=lambda: [])
    # where the second half of the model is saved.
    # If empty, the final FC layer of the first model is used.
    model_dirs2: List[str] = dataclasses.field(default_factory=lambda: [])

    model_dir: str = ""
    model_dir2: Optional[str] = ""

    eval_data: str = "test"  # whether to use test or val data
    batch_size: int = 16

    no_background: bool = False
    use_sigmoid: bool = False
    attr_sparsity: int = 1

    data_dir: str = "CUB_masked_class"  # directory to the data used for evaluation
    data_dir_raw: str = "CUB_processed"  # directory to the raw data
    n_attributes: int = 109
    image_dir: str = "images"  # test image folder to run inference on
    # file listing the (trained) model directory for each attribute group
    attribute_group: Optional[str] = None
    # whether to print out performance of individual atttributes
    feature_group_results: bool = False
    # Whether to correct with class- (if set) or instance- (if not set) level values
    class_level: bool = False

    # Which mode to use for correction. Only random actually implemented in original code
    mode: str = "random"
    n_trials: int = 1  # Number of trials to run, when mode is random
    n_groups: int = 28  # n. groups of attributes (28 enums with ~1 options = 312 attrs)
    multimodel: bool = True # whether to use the multimodel architecture


base_ind_tti_cfg = TTI_Config(
    n_trials=5,
    class_level=True,
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

seq_CtoY_cfg = Experiment(
    tag="seq_CtoY",
    exp="Sequential_CtoY",
    epochs=1000,
    lr=0.001,
    data_dir = "ConceptModel1__PredConcepts",
)

joint_cfg = Experiment(
    tag="joint",
    exp="Joint",
    epochs=1000,
    weighted_loss="multiple",
    attr_loss_weight=0.01,
    lr=0.001,
)

multiple_cfg = Experiment(
    multimodel=True,
    exp="Multimodel",
    epochs=1000,
    weighted_loss="multiple",
    lr=0.001,
    batch_size=24,
    attr_loss_weight=0.01,
    # multimodel specific
    n_models=3,
    freeze_post_models=True,
    reset_pre_models=True,
)