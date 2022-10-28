from datetime import datetime
from pathlib import Path
import random
from typing import List, Optional, Callable
import copy
import dataclasses
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import streamlit as st
import torch

import experiments as exps
from experiments import Experiment

from classes import Model, TrainResult, StepResult, Loss
from utils import _get_train_result_path, _load_train_result, _save_train_result, _get_average_loss

# With binary data and zero info, ideal prediction is always 0.5
ZERO_INFO_LOSS = 0.5**2
BINARY_COEFS_10 = [math.comb(10, x) for x in range(11)]


def main():
    st.title("Hidden info")

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
        # ax.set_yscale("log")
    fig.tight_layout()
    st.pyplot(fig)


def _hyperparameter_search(*experiments_iterable: Experiment) -> None:
    train_results = _run_experiments(*experiments_iterable)
    if not train_results:
        return

    # Compare results using the average p2 reconstruction loss of the last 10% of steps.
    df = []
    for train_result in train_results:
        last_step = max(step_result.step for step_result in train_result.step_results)
        reconstruction_loss_p2 = np.mean(
            [
                step_result.reconstruction_loss_p2
                for step_result in train_result.step_results
                if step_result.step >= last_step * 0.9
            ]
        )
        df.append(
            dict(tag=train_result.tag, reconstruction_loss_p2=reconstruction_loss_p2)
        )
    df = pd.DataFrame(df)
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="tag", y="reconstruction_loss_p2", ax=ax)
    st.pyplot(fig)


def _hyperparameter_search(*experiments_iterable: Experiment) -> None:
    train_results = _run_experiments(*experiments_iterable)
    if not train_results:
        return

    # Compare results using the average p2 reconstruction loss of the last 10% of steps.
    df = []
    for train_result in train_results:
        last_step = max(step_result.step for step_result in train_result.step_results)
        reconstruction_loss_p2 = np.mean(
            [
                step_result.reconstruction_loss_p2
                for step_result in train_result.step_results
                if step_result.step >= last_step * 0.9
            ]
        )
        df.append(
            dict(tag=train_result.tag, reconstruction_loss_p2=reconstruction_loss_p2)
        )
    df = pd.DataFrame(df)
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="tag", y="reconstruction_loss_p2", ax=ax)
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
        if force_retrain_models or _get_train_result_path(experiment.tag) is None:
            train_result = _train(experiment=experiment)
            _save_train_result(train_result)
        else:
            train_result = _load_train_result(experiment.tag)
        train_results.append(train_result)
        bar.progress((i + 1) / len(experiments))
    return train_results


def _train(experiment: Experiment) -> TrainResult:
    if experiment.seed is not None:
        torch.manual_seed(experiment.seed)
    if experiment.load_decoders_from_tag is not None:
        decoder_train_result = _load_train_result(experiment.load_decoders_from_tag)
        assert len(decoder_train_result.models) >= experiment.n_models
        dec_fn = lambda x: decoder_train_result.models[x].decoder
    else:
        dec_fn = lambda _: _create_decoder(
            latent_size=experiment.latent_size,
            hidden_size=experiment.hidden_size,
            vector_size=experiment.vector_size,
            use_class=experiment.use_class,
            n_hidden_layers=experiment.n_hidden_layers,
            activation_fn=experiment.activation_fn,
            dropout_prob=experiment.dropout_prob,
        )

    if experiment.load_encoders_from_tag is not None:
        encoder_train_result = _load_train_result(experiment.load_encoders_from_tag)
        assert len(encoder_train_result.models) >= experiment.n_models
        enc_fn = lambda x: encoder_train_result.models[x].encoder
    else:
        enc_fn = lambda _: _create_encoder(
            latent_size=experiment.latent_size,
            hidden_size=experiment.hidden_size,
            vector_size=experiment.vector_size,
            n_hidden_layers=experiment.n_hidden_layers,
            activation_fn=experiment.activation_fn,
        )

    models = [
        Model(
            encoder=enc_fn(ndx),
            decoder=dec_fn(ndx),
        )
        for ndx in range(experiment.n_models)
    ]

    if experiment.load_decoders_from_tag is not None:
        all_params = [[*model.encoder.parameters()] for model in models]
    elif experiment.load_encoders_from_tag is not None:
        all_params = [[*model.decoder.parameters()] for model in models]
    else:
        all_params = [
            [*model.encoder.parameters(), *model.decoder.parameters()]
            for model in models
        ]

    optimizer = torch.optim.Adam(
        list(itertools.chain.from_iterable(all_params)), lr=experiment.learning_rate
    )

    reconstruction_loss_fn: Callable  # I thought this was PYTHON
    representation_loss_fn: Callable
    target_latent_fn: Callable
    repr_loss_mask_fn: Callable

    if experiment.loss_quadrants == "all":
        repr_loss_mask_fn = lambda x: torch.ones(x.shape[0])
        repr_loss_scale = 1.0
    elif experiment.loss_quadrants == "bin_sum":
        repr_loss_mask_fn = _make_bin_sum_repr_mask(experiment.quadrant_threshold)
        repr_loss_scale = 2**10 / sum(
            BINARY_COEFS_10[: experiment.quadrant_threshold]
        )
    elif experiment.loss_quadrants == "bin_val":
        repr_loss_mask_fn = _make_bin_val_repr_mask(experiment.quadrant_threshold)
        repr_loss_scale = 2**10 / (2**10 - experiment.quadrant_threshold)
    else:
        raise ValueError(
            f"Loss quadrant must be 'all', 'bin_sum' or 'bin_val', got {experiment.loss_quadrants}."
        )

    if experiment.use_class:
        reconstruction_loss_fn = torch.nn.CrossEntropyLoss()
    else:
        reconstruction_loss_fn = torch.nn.MSELoss()

    if experiment.sparsity == 1:
        sparsity_fn = lambda x: x
        representation_loss_fn = _make_mse_loss_fn(
            repr_loss_mask_fn, target_repr_dim=experiment.preferred_rep_size
        )
    else:
        repr_sparsity_p = 1 - (1 / experiment.sparsity)
        sparsity_fn = torch.nn.Dropout(p=repr_sparsity_p)
        representation_loss_fn = _make_sparse_loss_fn(
            sparsity_fn=sparsity_fn,
            mask_fn=repr_loss_mask_fn,
            target_repr_dim=experiment.preferred_rep_size,
        )

    if experiment.loss_geometry == "simple":
        target_latent_fn = lambda x: x[:, : experiment.preferred_rep_size]
    elif experiment.loss_geometry == "diagonal":
        target_latent_fn = _make_diagonal_repr_fn(
            rep_size=experiment.preferred_rep_size
        )
    elif experiment.loss_geometry == "random_linear":
        target_latent_fn = _make_random_linear_repr_fn(
            rep_size=experiment.preferred_rep_size
        )
    else:
        raise ValueError(
            f"Loss geometry must be 'simple', 'diagonal' or 'random_linear', got {experiment.loss_geometry}."
        )

    bar = st.progress(0.0)
    step_results = []
    encoder_to_decoder_ndx = list(range(len(models)))
    for step in range(experiment.num_batches):
        # TODO: Delete this if?
        if experiment.dropout_prob is not None and step == 9000:
            pass
        if experiment.shuffle_decoders:
            random.shuffle(encoder_to_decoder_ndx)

        for encoder_ndx in range(len(models)):
            optimizer.zero_grad()
            decoder_ndx = encoder_to_decoder_ndx[encoder_ndx]
            encoder = models[encoder_ndx].encoder
            decoder = models[decoder_ndx].decoder

            vector = _generate_vector_batch(
                batch_size=experiment.batch_size,
                vector_size=experiment.vector_size,
                preferred_rep_size=experiment.preferred_rep_size,
                vector_p2_scale=experiment.vector_p2_scale,
            )
            if experiment.has_missing_knowledge:
                vector_input = torch.concat(
                    [
                        vector[:, : experiment.preferred_rep_size],
                        _generate_vector_batch(
                            batch_size=experiment.batch_size,
                            vector_size=experiment.vector_size,
                            preferred_rep_size=experiment.preferred_rep_size,
                            vector_p2_scale=experiment.vector_p2_scale,
                        )[:, experiment.preferred_rep_size :],
                    ],
                    dim=1,
                )
            else:
                vector_input = vector

            latent_repr = encoder(vector_input)
            if experiment.latent_masking:
                latent_repr = latent_repr[: experiment.preferred_rep_size]
                repr_mask = latent_use_mask_fn(latent_repr)
                # TODO: Unused variable?
                repr_use_loss = torch.mean(repr_mask)
            else:
                repr_use_loss = torch.Tensor(0)

            noise = torch.normal(
                mean=0, std=experiment.latent_noise_std, size=latent_repr.shape
            )
            target_latent = target_latent_fn(vector_input)
            if experiment.give_full_info:
                decoder_input = target_latent
            else:
                decoder_input = latent_repr + noise
            vector_reconstructed = decoder(decoder_input)
            if experiment.use_class:
                vector_reconstructed = vector_reconstructed.reshape(
                    experiment.batch_size, 2, experiment.vector_size
                )
                vector_target = vector.to(dtype=torch.long)
            else:
                vector_target = vector

            if experiment.give_full_info:
                sparsity_fn = torch.nn.Dropout(p=repr_sparsity_p)
                loss_fn = torch.nn.MSELoss(reduction="none")
                losses = sparsity_fn(loss_fn(vector_reconstructed, vector_target))
                mask = repr_loss_mask_fn(target_latent)
                masked_losses = (losses.T * mask).T
                reconstruction_loss = torch.mean(masked_losses) * repr_loss_scale

            else:
                reconstruction_loss = reconstruction_loss_fn(
                    vector_reconstructed, vector_target
                )
            representation_loss = representation_loss_fn(
                _input=latent_repr,
                target=target_latent_fn(vector),
            )
            # Scaling here to compensate for quadrant sparsity
            # TODO: Should we roll this into `experiment.representation_loss`?
            representation_loss *= repr_loss_scale
            loss = reconstruction_loss * experiment.reconstruction_loss_scale
            if experiment.representation_loss is not None:
                loss += experiment.representation_loss * representation_loss
            if experiment.l1_loss is not None:
                l1_loss = experiment.l1_loss * torch.norm(
                    latent_repr[experiment.preferred_rep_size :], 1
                )
                loss += l1_loss
            if experiment.l2_loss is not None:
                l2_loss = experiment.l2_loss * torch.norm(
                    latent_repr[experiment.preferred_rep_size :], 2
                )
                loss += l2_loss

            loss.backward()
            optimizer.step()
            if experiment.use_class:
                reconstruction_loss_p1 = reconstruction_loss_fn(
                    vector_reconstructed[:, :, : experiment.preferred_rep_size],
                    vector_target[:, : experiment.preferred_rep_size],
                )
                reconstruction_loss_p2 = reconstruction_loss_fn(
                    vector_reconstructed[:, :, experiment.preferred_rep_size :],
                    vector_target[:, experiment.preferred_rep_size :],
                )
            else:
                reconstruction_loss_p1 = reconstruction_loss_fn(
                    vector_reconstructed[:, : experiment.preferred_rep_size],
                    vector[:, : experiment.preferred_rep_size],
                )
                reconstruction_loss_p2 = reconstruction_loss_fn(
                    vector_reconstructed[:, experiment.preferred_rep_size :],
                    vector[:, experiment.preferred_rep_size :],
                )
            if step % 100 == 0:
                step_results.append(
                    StepResult(
                        tag=experiment.tag,
                        step=step,
                        encoder_ndx=encoder_ndx,
                        decoder_ndx=decoder_ndx,
                        total_loss=loss.item(),
                        reconstruction_loss=reconstruction_loss.item(),
                        representation_loss=representation_loss.item(),
                        reconstruction_loss_p1=reconstruction_loss_p1.item(),
                        reconstruction_loss_p2=reconstruction_loss_p2.item(),
                    )
                )

        bar.progress((step + 1) / experiment.num_batches)

    return TrainResult(experiment.tag, models, step_results)


def _create_encoder(
    vector_size: int,
    hidden_size: int,
    latent_size: int,
    n_hidden_layers: int,
    activation_fn: str,
    latent_mask: bool = False,
) -> torch.nn.Module:
    layers: List[torch.nn.Module] = []
    layers.append(torch.nn.Linear(vector_size, hidden_size))
    for _ in range(n_hidden_layers):
        layers.append(torch.nn.Linear(hidden_size, hidden_size))
        layers.append(_get_activation_fn(activation_fn))

    if latent_mask:
        output_size = latent_size * 2
    else:
        output_size = latent_size

    layers.append(torch.nn.Linear(hidden_size, output_size))
    return torch.nn.Sequential(*layers)


def _create_decoder(
    latent_size: int,
    hidden_size: int,
    vector_size: int,
    use_class: bool,
    n_hidden_layers: int,
    activation_fn: str,
    dropout_prob: Optional[float],
) -> torch.nn.Module:
    if use_class:
        output_size = vector_size * 2
    else:
        output_size = vector_size
    layers: List[torch.nn.Module] = []
    if dropout_prob is not None:
        layers.append(torch.nn.Dropout(p=dropout_prob))
    layers.append(torch.nn.Linear(latent_size, hidden_size))
    for _ in range(n_hidden_layers):
        layers.append(torch.nn.Linear(hidden_size, hidden_size))
        layers.append(_get_activation_fn(activation_fn))
    layers.append(torch.nn.Linear(hidden_size, output_size))
    return torch.nn.Sequential(*layers)


def _make_sparse_loss_fn(
    sparsity_fn: Callable, mask_fn: Callable, target_repr_dim: int
) -> Callable:
    loss_fn = torch.nn.MSELoss(reduction="none")

    def sparse_repr_loss_fn(_input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = sparsity_fn(loss_fn(_input, target))
        mask = mask_fn(target[:, :target_repr_dim])
        masked_losses = (losses.T * mask).T
        return torch.mean(masked_losses)

    return sparse_repr_loss_fn


def _make_mse_loss_fn(mask_fn: Callable, target_repr_dim: int) -> Callable:
    loss_fn = torch.nn.MSELoss(reduction="none")

    def mse_loss_fn(_input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = loss_fn(_input, target)
        mask = mask_fn(target[:, :target_repr_dim])
        masked_losses = (losses.T * mask).T
        return torch.mean(masked_losses)

    return mse_loss_fn


def _make_diagonal_repr_fn(rep_size: int) -> Callable:
    def diagonal_repr_target(_input: torch.Tensor) -> torch.Tensor:
        assert rep_size + 1 <= _input.shape[1]
        dir_1 = _input[:, :rep_size]
        # TODO: Wrap this so that we don't use N+1 variables?
        dir_2 = _input[:, 1 : rep_size + 1]
        repr_target = (dir_1 + dir_2) / np.sqrt(2)
        return repr_target

    return diagonal_repr_target


def _make_random_linear_repr_fn(rep_size: int) -> Callable:
    torch.manual_seed(0)
    proj_fn = torch.nn.Linear(rep_size, rep_size)
    scale_up = 5

    def random_linear_repr_target(_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return proj_fn(_input) * scale_up

    return random_linear_repr_target


def _make_bin_sum_repr_mask(threshold: int) -> Callable:
    def bin_sum_repr_mask(target: torch.Tensor) -> torch.Tensor:
        assert target.shape[1] == 10  # Quadrant options only work with 10dims of target
        mask = target.sum(dim=1) < threshold
        return mask

    return bin_sum_repr_mask


def _make_bin_val_repr_mask(threshold: int) -> Callable:
    bin_power_t = torch.Tensor([2**x for x in range(9, -1, -1)])

    def bin_val_repr_mask(target: torch.Tensor) -> torch.Tensor:
        assert target.shape[1] == 10  # Quadrant options only work with 10dims of target
        bin_vals = target * bin_power_t
        mask = bin_vals.sum(dim=1) < threshold
        return mask

    return bin_val_repr_mask


def _get_activation_fn(name: str) -> torch.nn.Module:
    if name == "relu":
        return torch.nn.ReLU()
    elif name == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise AssertionError(name)


def _generate_vector_batch(
    batch_size: int, vector_size: int, preferred_rep_size: int, vector_p2_scale: int
) -> torch.Tensor:
    # High is exclusive, so add one.
    p1_high = 1 + 1
    p2_high = vector_p2_scale + 1
    p1 = torch.randint(
        low=0, high=p1_high, size=(batch_size, preferred_rep_size)
    ).float()
    p2 = torch.randint(
        low=0, high=p2_high, size=(batch_size, vector_size - preferred_rep_size)
    ).float()
    return torch.concat([p1, p2], dim=1)

def load_results(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    
    _plot_results(results)

if __name__ == "__main__":
    main()
    
    # st.header("Showing big batch of sparase results")
    # results_path = Path("out/multiresult.pkl")
    # load_results(results_path)
    
