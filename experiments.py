from pathlib import Path
from typing import Optional
import dataclasses
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
    vector_p2_scale: int = 1
    repr_loss_coef: float = 5
    dropout_prob: Optional[float] = None
    l1_loss: Optional[float] = None
    l2_loss: Optional[float] = None
    activation_fn: torch.nn.Module = dataclasses.field(default_factory=torch.nn.ReLU)

    # Training setup
    num_batches: int = 10_000
    has_representation_loss: float = True
    has_missing_knowledge: bool = False
    end_to_end: bool = False
    n_models: int = 1
    load_encoders_from_tag: Optional[str] = None
    load_decoders_from_tag: Optional[str] = None


baseline = Experiment(
    tag="baseline",
    has_representation_loss=False,
    has_missing_knowledge=False,
)

repr_loss = Experiment(
    tag="representation_loss",
    has_representation_loss=True,
    has_missing_knowledge=False,
)

fuzzed = Experiment(
    tag="missing_knowledge",
    has_representation_loss=False,
    has_missing_knowledge=True,
)

dropout = Experiment(tag="dropout", dropout_prob=0.3)

freeze2 = Experiment(tag="freeze_2", end_to_end=True, n_models=2)
freeze4 = Experiment(tag="freeze_4", end_to_end=True, n_models=4)
freeze10 = Experiment(tag="freeze_10", end_to_end=True, n_models=10)

# Series of experiments that should be run in order
encoders_then_decoders = [
    Experiment(
        tag="prep3",
        n_models=3,
        use_class=False,
    ),
    Experiment(
        tag="fresh_enc_3",
        load_decoders_from_tag="prep3",
        end_to_end=True,
        n_models=3,
    ),
    Experiment(
        tag="fresh_dec_3",
        load_encoders_from_tag="fresh_enc_3",
        n_models=3,
        end_to_end=True,
    ),
]
decoders_then_encoders = [
    Experiment(
        tag="prep_",
        n_models=3,
        activation_fn=torch.nn.Sigmoid(),
        use_class=False,
    ),
    Experiment(
        tag="fresh_dec",
        activation_fn=torch.nn.Sigmoid(),
        load_encoders_from_tag="prep_",
        end_to_end=True,
        n_models=3,
    ),
    Experiment(
        tag="fresh_enc",
        activation_fn=torch.nn.Sigmoid(),
        load_decoders_from_tag="fresh_dec",
        n_models=3,
        end_to_end=True,
    ),
]

new_decoders = [
    Experiment(
        tag="prepare",
        n_models=3,
        activation_fn=torch.nn.Sigmoid(),
        use_class=False,
    ),
    Experiment(
        tag="fresh_dec",
        activation_fn=torch.nn.Sigmoid(),
        load_encoders_from_tag="prepare",
        end_to_end=True,
        n_models=3,
    ),
]

baseline_10_latent = Experiment(
    tag="10-latent-space",
    n_models=3,
    activation_fn=torch.nn.Sigmoid(),
    use_class=False,
    num_batches=10_000,
    latent_size=10,
    hidden_size=80,
    n_hidden_layers=1,
    has_representation_loss=True,
    repr_loss_coef=0.05,
)
