import dataclasses
from typing import Optional
from pathlib import Path


@dataclasses.dataclass
class Experiment:
    tag: str

    vector_size: int = 20
    latent_size: int = 10
    hidden_size: int = 20
    num_batches: int = 20_000
    batch_size: int = 32
    num_iterations: int = 1
    vector_p2_scale: int = 3
    dropout_p: float = 0.3
    repr_loss_coef: int = 5

    has_representation_loss: float = True
    has_missing_knowledge: bool = False
    dropout: bool = False
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
)

dropout = Experiment(tag="dropout", dropout=True)

freeze2 = Experiment(tag="freeze_2", end_to_end=True, n_models=2)
freeze4 = Experiment(tag="freeze_4", end_to_end=True, n_models=4)
freeze10 = Experiment(tag="freeze_10", end_to_end=True, n_models=10)

# Series of experiments that should be run in order
prep_decoders3 = Experiment(
    tag="prep3",
    n_models=3,
    save_model=Path("./out/store/decoders.pickle"),
)
fresh_encoders3 = Experiment(
    tag="fresh3",
    load_decoder=True,
    n_models=3,
    save_model=Path("./out/store/encoders.pickle"),
)
fresh_decoder = Experiment(tag="fresh_dec", load_encoder=True, n_models=3)
