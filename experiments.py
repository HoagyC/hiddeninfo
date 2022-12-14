import dataclasses
from typing import List

from classes import Experiment

baseline = Experiment(
    tag="baseline",
    representation_loss=None,
    has_missing_knowledge=False,
)

repr_loss = Experiment(
    tag="representation_loss",
    has_missing_knowledge=False,
)

fuzzed = Experiment(
    tag="missing_knowledge",
    representation_loss=None,
    has_missing_knowledge=True,
)

base = Experiment(
    tag="base",
    n_models=8,
    activation_fn="sigmoid",
    use_class=False,
    num_batches=2_000,
    latent_size=10,
    hidden_size=80,
    n_hidden_layers=1,
    # Ideally, we want the representation loss to be as high as possible, conditioned on the
    # autoencoders still exhibiting the "hidden info" behaviour.
    # TODO: Experiment with making this number larger.
    representation_loss=1,
)

dropout = Experiment(tag="dropout", dropout_prob=0.3)

freeze2 = Experiment(tag="freeze_2", shuffle_decoders=True, n_models=2)
freeze4 = Experiment(tag="freeze_4", shuffle_decoders=True, n_models=4)
freeze10 = Experiment(tag="freeze_10", shuffle_decoders=True, n_models=10)

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
        shuffle_decoders=True,
        n_models=3,
    ),
    Experiment(
        tag="fresh_dec_3",
        load_encoders_from_tag="fresh_enc_3",
        n_models=3,
        shuffle_decoders=True,
    ),
]
decoders_then_encoders = [
    Experiment(
        tag="prep_",
        n_models=3,
        activation_fn="sigmoid",
        use_class=False,
    ),
    Experiment(
        tag="fresh_dec",
        activation_fn="sigmoid",
        load_encoders_from_tag="prep_",
        shuffle_decoders=True,
        n_models=3,
    ),
    Experiment(
        tag="fresh_enc",
        activation_fn="sigmoid",
        load_decoders_from_tag="fresh_dec",
        n_models=3,
        shuffle_decoders=True,
    ),
]

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

baseline_10_latent = Experiment(
    tag="latent10",
    n_models=3,
    activation_fn="sigmoid",
    use_class=False,
    num_batches=10_000,
    latent_size=10,
    hidden_size=80,
    n_hidden_layers=1,
    representation_loss=0.05,
)
baseline_10_latent_noisy = dataclasses.replace(
    baseline_10_latent,
    tag="latent10-noisy",
    latent_noise_std=0.5,
)


def make_sparse_exps(
    base_experiment: Experiment = repr_loss, n_powers=6
) -> List[Experiment]:
    tag = "sparse"
    sparsity_levels = [10**x for x in range(n_powers)]
    return [
        dataclasses.replace(base_experiment, tag=f"{tag}{sparsity}", sparsity=sparsity)
        for sparsity in sparsity_levels
    ]


def make_retrain_enc_experiments(base_experiment: Experiment) -> List[Experiment]:
    tag = base_experiment.tag
    return [
        dataclasses.replace(base_experiment, tag=f"{tag}_init"),
        dataclasses.replace(
            base_experiment,
            tag=f"{tag}_retrain-encoders",
            load_decoders_from_tag=f"{tag}_init",
            shuffle_decoders=True,
        ),
    ]


def make_retrain_dec_experiments(base_experiment: Experiment) -> List[Experiment]:
    tag = base_experiment.tag
    return [
        dataclasses.replace(base_experiment, tag=f"{tag}_init"),
        dataclasses.replace(
            base_experiment,
            tag=f"{tag}_retrain-decoders",
            load_encoders_from_tag=f"{tag}_init",
            shuffle_decoders=True,
        ),
    ]


def make_retrain_enc_dec_experiments(base_experiment: Experiment) -> List[Experiment]:
    tag = base_experiment.tag
    return [
        *make_retrain_enc_experiments(base_experiment),
        dataclasses.replace(
            base_experiment,
            tag=f"{tag}_retrain-decoders",
            load_encoders_from_tag=f"{tag}_retrain-encoders",
            shuffle_decoders=True,
        ),
    ]


def make_retrain_dec_enc_experiments(base_experiment: Experiment) -> List[Experiment]:
    tag = base_experiment.tag
    return [
        *make_retrain_dec_experiments(base_experiment),
        dataclasses.replace(
            base_experiment,
            tag=f"{tag}_retrain-encoders",
            load_decoders_from_tag=f"{tag}_retrain-decoders",
            shuffle_decoders=True,
        ),
    ]
