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
    repr_loss_coef: int = 5
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
    load_encoder: bool = False
    encoder_loc: Path = Path("./out/store/encoders.pickle")
    load_decoder: bool = False
    decoder_loc: Path = Path("./out/store/decoders.pickle")
    save_model: Optional[Path] = None


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
    save_model=Path("./out/store/fuzzed_baseline"),
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
        save_model=Path("./out/store/decoders"),
        use_class=False,
    ),
    Experiment(
        tag="fresh_enc_3",
        load_decoder=True,
        decoder_loc=Path("./out/store/decoders.pickle"),
        end_to_end=True,
        n_models=3,
        save_model=Path("./out/store/encoders.pickle"),
    ),
    Experiment(
        tag="fresh_dec_3",
        load_encoder=True,
        encoder_loc=Path("./out/store/encoders.pickle"),
        n_models=3,
        end_to_end=True,
    ),
]
decoders_then_encoders = [
    Experiment(
        tag="prep_",
        n_models=3,
        activation_fn=torch.nn.Sigmoid(),
        save_model=Path("./out/store/encoders"),
        use_class=False,
    ),
    Experiment(
        tag="fresh_dec",
        activation_fn=torch.nn.Sigmoid(),
        load_encoder=True,
        encoder_loc=Path("./out/store/encoders"),
        end_to_end=True,
        n_models=3,
        save_model=Path("./out/store/decoders"),
    ),
    Experiment(
        tag="fresh_enc",
        activation_fn=torch.nn.Sigmoid(),
        load_decoder=True,
        decoder_loc=Path("./out/store/decoders"),
        n_models=3,
        end_to_end=True,
    ),
]

new_decoders = [
    Experiment(
        tag="prepare",
        n_models=3,
        activation_fn=torch.nn.Sigmoid(),
        save_model=Path("./out/store/encoders"),
        use_class=False,
    ),
    Experiment(
        tag="fresh_dec",
        activation_fn=torch.nn.Sigmoid(),
        load_encoder=True,
        encoder_loc=Path("./out/store/encoders"),
        end_to_end=True,
        n_models=3,
    ),
]
