import dataclasses
from typing import List
from typing import Optional

import torch


@dataclasses.dataclass
class Experiment:
    tag: str

    use_class: bool = False

    # Model setup
    vector_size: int = 20
    latent_size: int = 25
    preferred_rep_size: int = 10
    n_hidden_layers: int = 0
    hidden_size: int = 30
    batch_size: int = 32
    representation_loss: Optional[float] = 5
    reconstruction_loss_scale: float = 1
    dropout_prob: Optional[float] = None
    l1_loss: Optional[float] = None
    l2_loss: Optional[float] = None
    # TODO: Use enums for this & other strings?
    activation_fn: str = "relu"

    # Training setup
    num_batches: int = 10_000
    seed: Optional[int] = None
    has_missing_knowledge: bool = False
    shuffle_decoders: bool = False
    n_models: int = 1
    load_encoders_from_tag: Optional[str] = None
    load_decoders_from_tag: Optional[str] = None
    give_full_info: bool = False
    learning_rate: float = 1e-03
    use_multiprocess: bool = False

    # Representation loss options
    loss_geometry: str = "simple"
    loss_quadrants: str = "all"
    # Useful range is [1, 1023] for 'bin_val' and [1, 9] for 'bin_sum'
    quadrant_threshold: int = 0
    sparsity: int = 1  # repr_loss scaled up by sparsity, applied every 1/sparsity
    latent_noise_std: float = 0
    latent_masking: bool = False
    latent_masking_incentive: float = 0.1


@dataclasses.dataclass
class Model:
    encoder: torch.nn.Module
    decoder: torch.nn.Module


@dataclasses.dataclass
class Loss:
    total_loss: float
    reconstruction_loss: float
    representation_loss: float
    # The reconstruction losses for the first & second halves of the vector.
    reconstruction_loss_p1: float
    reconstruction_loss_p2: float


@dataclasses.dataclass
class StepResult:
    tag: str
    step: int
    total_loss: float
    reconstruction_loss: float
    representation_loss: float
    # The reconstruction losses for the first & second halves of the vector.
    reconstruction_loss_p1: float
    reconstruction_loss_p2: float

    # Optional for backwards compatibility
    encoder_ndx: Optional[int] = None
    decoder_ndx: Optional[int] = None


@dataclasses.dataclass
class TrainResult:
    tag: str
    models: List[Model]
    step_results: List[StepResult]
    experiment: Optional[Experiment] = None
    # validation_result: StepResult
