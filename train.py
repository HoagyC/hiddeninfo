import dataclasses
import itertools
import math
import multiprocessing as mp
import random
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import streamlit as st
import torch

from classes import Model
from classes import StepResult
from classes import TrainResult
from experiments import Experiment
from utils import _load_train_result

BINARY_COEFS_10 = [math.comb(10, x) for x in range(11)]
NUM_PROCESSES = 4


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

    if (
        experiment.n_models > 1
        and not experiment.shuffle_decoders
        and experiment.use_multiprocess
    ):
        result = _run_separate_models(experiment, models)
    else:
        result = _run_models(experiment, models)

    return result


def _combine_results(results=List[TrainResult]) -> TrainResult:
    models = [r.models for r in results]
    tag = results[0].tag
    step_results = []
    for ndx, r in enumerate(results):
        step_results += [
            dataclasses.replace(sr, encoder_ndx=ndx, decoder_ndx=ndx)
            for sr in r.step_results
        ]

    return TrainResult(tag=tag, models=models, step_results=step_results)


def _run_separate_models(experiment: Experiment, models: List[Model]):
    args_iter = [{"experiment": experiment, "models": [model]} for model in models]
    results = []
    with mp.Pool(NUM_PROCESSES) as pool:
        results_iter = pool.imap_unordered(_run_models_helper, args_iter)
        for r in results_iter:
            results.append(r)
    result = _combine_results(results)
    return result


def _run_models_helper(args: Dict) -> TrainResult:
    return _run_models(**args)


def _run_models(experiment: Experiment, models: List[Model]) -> TrainResult:

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

    repr_sparsity_p = 1 - (1 / experiment.sparsity)
    if experiment.sparsity == 1:
        sparsity_fn = lambda x: x
        representation_loss_fn = _make_mse_loss_fn(
            repr_loss_mask_fn, target_repr_dim=experiment.preferred_rep_size
        )
    else:
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
