import dataclasses
from typing import List, Optional

import torch


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
    # validation_result: StepResult
