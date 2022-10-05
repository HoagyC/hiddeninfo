import dataclasses
from typing import Optional
from pathlib import Path


@dataclasses.dataclass
class Experiment:
    tag: str
    has_representation_loss: float = True
    has_missing_knowledge: bool = False
    dropout: bool = False
    end_to_end: bool = False
    n_models: int = 1
    prep_decoders: bool = True
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
    prep_decoders=True,
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
