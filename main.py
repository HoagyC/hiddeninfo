import copy
import dataclasses
import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import experiments as exps
import train
from classes import TrainResult
from experiments import Experiment
from experiments import base
from utils import get_train_result_path
from utils import load_train_result
from utils import save_train_result

# With binary data and zero info, ideal prediction is always 0.5
ZERO_INFO_LOSS = 0.5**2


def main():
    st.title("Hidden info")

    l1 = dataclasses.replace(base, tag="l1", l1_loss=1e-5, n_models=1)
    l2 = dataclasses.replace(base, tag="l2", l2_loss=1e-2, n_models=1)
    dropout = dataclasses.replace(base, tag="dropout", dropout_prob=0.5, n_models=1)
    noisy = dataclasses.replace(base, tag="noisy", latent_noise_std=0.1, n_models=1)
    perms = dataclasses.replace(base, tag="perms", shuffle_decoders=True)
    retrain_dec = dataclasses.replace(
        base,
        tag="retrain-dec",
        load_encoders_from_tag=base.tag,
        shuffle_decoders=True,
    )
    retrain_enc = dataclasses.replace(
        base,
        tag="retrain-enc",
        load_decoders_from_tag=base.tag,
        shuffle_decoders=True,
        # Don't use representation loss when retraining the encoders.
        # This is kinda cheating: if the problem of retraining the encoders is too hard, then adding
        # the representation loss gives the model a *really* easy way to fit the the decoder for p1,
        # leaving it struggling for p2.
        representation_loss=0,
    )
    retrain_enc_with_repr = dataclasses.replace(
        base,
        tag="retrain_enc_repr",
        load_decoders_from_tag=base.tag,
        shuffle_decoders=True,
        # Don't use representation loss when retraining the encoders.
        # This is kinda cheating: if the problem of retraining the encoders is too hard, then adding
        # the representation loss gives the model a *really* easy way to fit the the decoder for p1,
        # leaving it struggling for p2.
    )

    diagonal_base = dataclasses.replace(
        base,
        tag="diagonal_base",
        loss_geometry="diagonal",
        latent_size=12,
    )
    diagonal_retrain_enc = dataclasses.replace(
        base,
        tag="diagonal_retrain_enc",
        loss_geometry="diagonal",
        shuffle_decoders=True,
        load_encoders_from_tag=diagonal_base.tag,
        latent_size=12,
        representation_loss=0,
    )

    diagonal_retrain_dec = dataclasses.replace(
        base,
        tag="diagonal_retrain_dec",
        loss_geometry="diagonal",
        shuffle_decoders=True,
        load_decoders_from_tag=diagonal_retrain_enc.tag,
        latent_size=12,
        representation_loss=0,
    )

    rand_lin_base = dataclasses.replace(
        base, tag="rand_lin_base", loss_geometry="random_linear"
    )
    rand_lin_retrain_enc = dataclasses.replace(
        retrain_dec,
        tag="rand_lin_retrain_enc",
        loss_geometry="random_linear",
        shuffle_decoders=True,
        load_encoders_from_tag=rand_lin_base.tag,
        representation_loss=0,
    )

    bin_sum_quads_5 = dataclasses.replace(
        base,
        tag="bin_sum_quads_5",
        loss_quadrants="bin_sum",
        quadrant_threshold=5,
        n_models=3,
        num_batches=30_000,
    )
    bin_sum_quads_5_enc = dataclasses.replace(
        bin_sum_quads_5,
        tag="bin_sum_quads_5_enc",
        load_decoders_from_tag=bin_sum_quads_5.tag,
        shuffle_decoders=True,
        quadrant_threshold=5,
        n_models=3,
    )

    bin_sum_quads_7 = dataclasses.replace(
        bin_sum_quads_5, tag="bin_sum_quads_7", quadrant_threshold=7, num_batches=30_000
    )
    bin_sum_quads_7_enc = dataclasses.replace(
        bin_sum_quads_5_enc,
        tag="bin_sum_quads_7_enc",
        quadrant_threshold=7,
        load_decoders_from_tag=bin_sum_quads_7.tag,
    )
    bin_sum_quads_9 = dataclasses.replace(
        bin_sum_quads_5, tag="bin_sum_quads_9", quadrant_threshold=9, num_batches=10_000
    )
    bin_sum_quads_9_enc = dataclasses.replace(
        bin_sum_quads_5_enc,
        tag="bin_sum_quads_9_enc",
        quadrant_threshold=9,
        load_decoders_from_tag=bin_sum_quads_9.tag,
        num_batches=10_000,
    )

    sparse_general = dataclasses.replace(
        base,
        tag="sparse_general",
        loss_quadrants="bin_sum",
        quadrant_threshold=4,
        num_batches=30_000,
        sparsity=10,
    )
    sparse_general_enc = dataclasses.replace(
        sparse_general,
        tag="sparse_general_enc",
        load_decoders_from_tag=sparse_general.tag,
        shuffle_decoders=True,
    )
    sparse_general_dec = dataclasses.replace(
        sparse_general,
        tag="sparse_general_dec",
        load_encoders_from_tag=sparse_general.tag,
        shuffle_decoders=True,
    )

    sequential_encoder = dataclasses.replace(
        base, tag="sequential_enc", reconstruction_loss_scale=0, num_batches=5000
    )
    sequential_decoder = dataclasses.replace(
        base, tag="sequential_dec", give_full_info=True, num_batches=5000
    )
    sequential_test = dataclasses.replace(
        base,
        tag="sequential_test",
        load_decoders_from_tag=sequential_decoder.tag,
        load_encoders_from_tag=sequential_encoder.tag,
        num_batches=10000,
    )
    seq_sparse_decoder = dataclasses.replace(
        base,
        tag="seq_sparse_dec",
        loss_quadrants="bin_sum",
        quadrant_threshold=4,
        sparsity=10,
        give_full_info=True,
        num_batches=10000,
    )

    seq_sparse_encoder = dataclasses.replace(
        base,
        tag="seq_sparse_enc",
        loss_quadrants="bin_sum",
        quadrant_threshold=4,
        sparsity=10,
        reconstruction_loss_scale=0,
        num_batches=10000,
    )
    seq_sparse_test = dataclasses.replace(
        base,
        tag="sequential_test",
        load_decoders_from_tag=seq_sparse_decoder.tag,
        load_encoders_from_tag=seq_sparse_encoder.tag,
        num_batches=2000,
    )

    seed_test1 = dataclasses.replace(
        base, tag="seedtest_1", n_models=1, num_batches=1000, seed=1
    )
    seed_test2 = dataclasses.replace(seed_test1, tag="seedtest_2")
    # One thing I've found is that it's hard to retrain the encoders. My hypothesis is that, since
    # the decoder is trying to find some hidden info in the latent embedding, it's *really*
    # sensitive around 0 and 1 values. This makes the loss landscape really difficult for GD to
    # traverse around these values, leading to a something has poor reconstruction performance.
    # I've not observed this effect recently, but noting here for posterity.
    # retrain_enc_noisy = dataclasses.replace(
    #     noisy,
    #     tag="retrain-enc-noisy",
    #     load_decoders_from_tag=noisy.tag,
    #     shuffle_decoders=True,
    #     representation_loss=0,
    # )
    st.header("Multiprocessing test")
    multiproc_test = dataclasses.replace(
        base, tag="multitest", use_multiprocess=True, n_models=4, num_batches=1000
    )
    multiproc_comp = dataclasses.replace(
        multiproc_test, tag="multicomp", use_multiprocess=False
    )
    _display_experiments(multiproc_test, multiproc_comp)

    st.header("Binary sum quads")
    _display_experiments(
        bin_sum_quads_5,
        bin_sum_quads_5_enc,
        bin_sum_quads_7,
        bin_sum_quads_7_enc,
        bin_sum_quads_9,
        bin_sum_quads_9_enc,
        base,
        retrain_enc,
    )

    st.header("Baseline")
    _display_experiments(base)

    st.header("Regularisation strategies")
    _display_experiments(l1, l2)

    st.header("Sequential")
    _display_experiments(sequential_encoder, sequential_decoder, sequential_test)

    st.header("Sequential sparse")
    _display_experiments(seq_sparse_encoder, seq_sparse_decoder, seq_sparse_test)

    st.header("E2E sparse")
    _display_experiments(sparse_general, sparse_general_enc, sparse_general_dec)

    st.header("seq and e2e sparse")
    _display_experiments(
        seq_sparse_test, seq_sparse_decoder, seq_sparse_encoder, sparse_general_dec
    )

    st.header("Dropout strategy")
    st.write("TODO: Get results outside of eval mode.")
    _display_experiments(dropout)

    st.header("Noise strategy")
    _display_experiments(noisy)

    st.header("Testing alternate target latents")
    _display_experiments(diagonal_base, diagonal_retrain_enc, diagonal_retrain_dec)
    _display_experiments(rand_lin_base, rand_lin_retrain_enc)

    st.header("Increasing sparsity")
    sparsity_exps = exps.make_sparse_exps()
    _display_experiments(*sparsity_exps)

    st.header("Random permutations")
    _display_experiments(perms)

    st.header("Retrain decoders + random permutations")
    _display_experiments(base, retrain_dec)

    st.header("Just retrain encoders")
    _display_experiments(retrain_enc_with_repr)

    st.header("Retrain encoders + random permutations + sparsity")
    _display_experiments(
        base, retrain_enc_with_repr, retrain_dec, sparse_general_dec, sparse_general_enc
    )

    st.header("All strategies")
    _display_experiments(base, l1, l2, dropout, noisy, perms, retrain_dec, retrain_enc)

    st.header("Experimenting with different numbers of models")
    for n_models in [2, 4, 8, 16]:
        st.subheader(f"Training with {n_models} models")
        experiments = copy.deepcopy(exps.new_decoders)
        for exp in experiments:
            suffix = f"_{n_models}models"
            exp.n_models = n_models
            exp.tag += suffix
            if exp.load_decoders_from_tag is not None:
                exp.load_decoders_from_tag += suffix
            if exp.load_encoders_from_tag is not None:
                exp.load_encoders_from_tag += suffix
        _display_experiments(*experiments)

    st.header("Test setting seed")
    _display_experiments(seed_test1, seed_test2)

    st.header("Tuning number of models")
    n_models_options = [2, 4, 8]
    tuning_n_models_base = dataclasses.replace(
        base, tag="tuning_n_models_base", n_models=max(n_models_options)
    )
    _display_experiments(tuning_n_models_base)
    _display_experiments(
        *(
            dataclasses.replace(
                tuning_n_models_base,
                tag=f"n_models={n_models}",
                n_models=n_models,
                load_encoders_from_tag=tuning_n_models_base.tag,
                shuffle_decoders=True,
                representation_loss=0,
            )
            for n_models in n_models_options
        )
    )


def _display_experiments(*experiments_iterable: Experiment) -> None:
    train_results = _run_experiments(*experiments_iterable)
    if not train_results:
        return

    _plot_results(train_results)


def _plot_results(train_results: List[TrainResult]) -> None:
    df = pd.DataFrame(
        dataclasses.asdict(result)
        for train_result in train_results
        for result in train_result.step_results
    )

    losses = [
        "representation_loss",
        "reconstruction_loss_p1",
        "reconstruction_loss_p2",
    ]
    fig, axs = plt.subplots(1, len(losses), figsize=(5 * len(losses), 5))
    for loss_name, ax in zip(losses, axs):
        sns.lineplot(data=df, x="step", y=loss_name, hue="tag", ax=ax)
        if "reconstruction" in loss_name:
            sns.lineplot(
                x=[0, max(df.step)],
                y=[ZERO_INFO_LOSS, ZERO_INFO_LOSS],
                label="zero info loss",
                linestyle="dashed",
                ax=ax,
            )
        ax.set_title(loss_name)
        ax.set_yscale("linear")
        ax.set_ylim(([0, 0.4]))
    fig.tight_layout()
    st.pyplot(fig)


def _run_experiments(*experiments_iterable: Experiment) -> List[TrainResult]:
    experiments: List[Experiment] = list(experiments_iterable)
    tags = [experiment.tag for experiment in experiments]
    assert len(tags) == len(set(tags)), f"Found duplicate tags: {tags}"
    if not st.checkbox(f"Run?", value=False, key=str((tags, "run"))):
        return []
    force_retrain_models = st.checkbox(
        "Force retrain models?", value=False, key=str((tags, "retrain"))
    )
    st.write(pd.DataFrame(dataclasses.asdict(experiment) for experiment in experiments))

    bar = st.progress(0.0)
    train_results: List[TrainResult] = []
    for i, experiment in enumerate(experiments):
        if force_retrain_models or get_train_result_path(experiment.tag) is None:
            train_result = train.train(experiment=experiment)
            save_train_result(train_result)
        else:
            train_result = load_train_result(experiment.tag)
        train_results.append(train_result)
        bar.progress((i + 1) / len(experiments))
    return train_results


def load_results(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    _plot_results(results)


if __name__ == "__main__":
    main()
