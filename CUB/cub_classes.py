import dataclasses
from typing import Optional, List

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
class BaseConf:
    use_attr: bool
    no_img: bool
    image_dir: str
    n_class_attr: int
    attr_sparsity: int
    batch_size: int


@dataclasses.dataclass
class Experiment(BaseConf):
    tag: str = "basic"
    exp: str = "multimodel"
    seed: int = 0

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
    use_relu: bool = True
    use_sigmoid: bool = False
    pretrained: bool = False
    use_aux: bool = False

    # Training
    epochs: int = 100
    end2end: bool = True
    optimizer: str = "SGD"
    scheduler_step: int = 1000
    attr_loss_weight: float = 1.0
    lr: float = 1e-03
    weight_decay: float = 5e-5
    attr_sparsity: int = 1

    # Legacy
    connect_CY: bool = False
    resampling: bool = False
    batch_size: int = 32

    freeze: bool = False
    use_attr: bool = False
    no_img: bool = False
    bottleneck: bool = False
    weighted_loss: str = ""
    uncertain_labels: bool = False
    normalize_loss: bool = False

    # Shuffling
    shuffle_models: bool = False
    freeze_post_models: bool = False
    freeze_pre_models: bool = False
    reset_pre_models: bool = False
    reset_post_models: bool = False
    n_models: int = 4

    # Can predict whethe trait is visible as a third class, n_class_attr=3
    n_class_attr: int = 2
    three_class: bool = n_class_attr == 3
    quick: bool = False


@dataclasses.dataclass
class TTI_Config(BaseConf):
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

    # How relevant for tti?? - because they're used to load up the dataset
    use_attr: bool = (
        False  # whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)
    )
    no_img: bool = False  # if included, only use attributes (and not raw imgs) for class prediction
    bottleneck: bool = False
    no_background: bool = False
    n_class_attr: int = 2
    use_relu: bool = False
    use_sigmoid: bool = False
    connect_CY: bool = (
        False  # Add the c->y fully connected layer into the inception model??
    )
    attr_sparsity: int = 1

    data_dir: str = "CUB_masked_class"  # directory to the data used for evaluation
    data_dir2: str = "CUB_processed"  # directory to the raw data
    n_attributes: int = 109
    image_dir: str = "images"  # test image folder to run inference on
    # file listing the (trained) model directory for each attribute group
    attribute_group: Optional[str] = None
    # whether to print out performance of individual atttributes
    feature_group_results: bool = False
    # Whether to correct with class- (if set) or instance- (if not set) level values
    class_level: bool = False
    use_invisible: bool = False

    # Which mode to use for correction. Only random actually implemented in original code
    mode: str = "random"
    n_trials: int = 1  # Number of trials to run, when mode is random
    n_groups: int = 28  # n. groups of attributes (28 enums with ~1 options = 312 attrs)
    multimodel: bool = True # whether to use the multimodel architecture
