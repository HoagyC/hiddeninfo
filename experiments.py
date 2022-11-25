import dataclasses
from typing import List

from classes import Experiment

repr_loss = Experiment(
    tag="representation_loss",
)
new_decoders = [
    Experiment(
        tag="prepare",
        n_models=3,
        activation_fn="sigmoid",
        use_class=False,
    ),
    Experiment(
        tag="fresh_dec",
        activation_fn="sigmoid",
        load_encoders_from_tag="prepare",
        shuffle_decoders=True,
        n_models=3,
    ),
]


def make_sparse_exps(
    base_experiment: Experiment = repr_loss, n_powers=6
) -> List[Experiment]:
    tag = "sparse"
    sparsity_levels = [10**x for x in range(n_powers)]
    return [
        dataclasses.replace(base_experiment, tag=f"{tag}{sparsity}", sparsity=sparsity)
        for sparsity in sparsity_levels
    ]
