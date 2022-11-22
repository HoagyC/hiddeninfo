import dataclasses

N_ATTRIBUTES = 312
N_CLASSES = 200


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
    seed: int = 0

    # Data
    log_dir: str = "out"
    data_dir: str = "CUB_processed"
    image_dir: str = "images"
    save_step: int = 10

    # Model
    multimodel: bool = True
    n_attributes: int = N_ATTRIBUTES
    num_classes: int = N_CLASSES
    expand_dim: int = 500
    use_relu: bool = True
    use_sigmoid: bool = False
    pretrained: bool = True
    use_aux: bool = True

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
    use_attr: bool = True
    no_img: bool = False
    bottleneck: bool = True
    weighted_loss: bool = False
    uncertain_labels: bool = True

    # Shuffling
    shuffle_models: bool = False
    freeze_post_models: bool = False
    freeze_pre_models: bool = False
    reset_pre_models: bool = False
    reset_post_models: bool = False
    n_models: int = 1

    # Can predict whethe trait is visible as a third class, n_class_attr=3
    n_class_attr: int = 2
    three_class: bool = n_class_attr == 3
