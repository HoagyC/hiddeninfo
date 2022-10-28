import dataclasses
import itertools
import random

from typing import Optional, List

import numpy as np
import torch
import streamlit as st

from CUB.models import ModelXtoC, ModelCtoY
from CUB.dataset import load_data

from utils import _load_train_result
from classes import Model, TrainResult, Loss, StepResult

N_ATTRIBUTES = 312
N_CLASSES = 200


@dataclasses.dataclass
class CUB_Loss:
    total_loss: float
    reconstruction_loss: float
    representation_loss: float


@dataclasses.dataclass
class EpochResult:
    tag: str
    epoch: int
    total_loss: float
    reconstruction_loss: float
    representation_loss: float


@dataclasses.dataclass
class CUB_Experiment:
    tag: str
    seed: Optional[int] = None

    representation_loss: Optional[float] = 1.0

    load_decoders_from_tag: Optional[str] = None
    load_encoders_from_tag: Optional[str] = None
    shuffle_decoders: bool = False

    freeze_decoder: bool = False
    n_models: int = 1
    num_epochs: int = 10_000

    hidden_dim_dec: int = 800
    sparsity: int = 1


@dataclasses.dataclass
class CUB_TrainResult:
    tag: str
    models: List[Model]
    step_results: List[EpochResult]


def _get_average_loss_cub(losses: List[CUB_Loss]) -> CUB_Loss:
    if len(losses) == 1:
        return losses[0]
    else:
        return CUB_Loss(
            float(np.mean([l.total_loss for l in losses])),
            float(np.mean([l.reconstruction_loss for l in losses])),
            float(np.mean([l.representation_loss for l in losses])),
        )


def _train(experiment: CUB_Experiment) -> CUB_TrainResult:
    if experiment.seed is not None:
        torch.manual_seed(experiment.seed)
    if experiment.load_decoders_from_tag is not None:
        decoder_train_result = _load_train_result(experiment.load_decoders_from_tag)
        assert len(decoder_train_result.models) >= experiment.n_models
        dec_fn = lambda x: decoder_train_result.models[x].decoder
    else:
        dec_fn = lambda _: ModelCtoY(
            n_attributes=N_ATTRIBUTES,
            num_classes=N_CLASSES,
            hidden_dim=experiment.hidden_dim_dec,
        )

    if experiment.load_encoders_from_tag is not None:
        encoder_train_result = _load_train_result(experiment.load_encoders_from_tag)
        assert len(encoder_train_result.models) >= experiment.n_models
        enc_fn = lambda x: encoder_train_result.models[x].encoder
    else:
        enc_fn = lambda _: ModelXtoC(
            pretrained=True,
            freeze=experiment.freeze_decoder,
            num_classes=N_CLASSES,
            use_aux=False,  # TO INVESTIGATE
            n_attributes=N_ATTRIBUTES,
            expand_dim=False,
            three_class=False,
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

    optimizer = torch.optim.Adam(list(itertools.chain.from_iterable(all_params)))

    reconstruction_loss_fn = torch.nn.CrossEntropyLoss()
    representation_loss_fns = [torch.nn.CrossEntropyLoss() for _ in range(N_ATTRIBUTES)]

    bar = st.progress(0.0)
    epoch_results: List[EpochResult] = []
    encoder_to_decoder_ndx = list(range(len(models)))

    train_data_path = "CUB_processed/train.pkl"
    data_loader = load_data(
        pkl_paths=[train_data_path], use_attr=True, no_img=False, batch_size=2
    )

    for epoch_ndx in range(experiment.num_epochs):
        losses = []
        for batch_ndx, data in enumerate(data_loader):
            encoder_ndx = random.randint(0, len(models) - 1)
            if experiment.shuffle_decoders:
                decoder_ndx = random.randint(0, len(models) - 1)
            else:
                decoder_ndx = encoder_ndx

            inputs, labels, attr_labels = data

            print(inputs, labels, attr_labels)

            optimizer.zero_grad()
            encoder = models[encoder_ndx].encoder
            decoder = models[decoder_ndx].decoder

            attrs_reconstructed = encoder(inputs)
            print(attrs_reconstructed, len(attrs_reconstructed))
            labels_reconstructed = decoder(attrs_reconstructed)

            reconstruction_loss = reconstruction_loss_fn(labels_reconstructed, labels)
            representation_losses = []
            for i, attr_result in enumerate(attrs_reconstructed):
                representation_losses.append(representation_loss_fns[i](
                    attr_result.squeeze(),
                    attr_labels,
                )

            loss = reconstruction_loss
            if experiment.representation_loss is not None:
                loss += experiment.representation_loss * representation_loss

            loss.backward()
            optimizer.step()

            losses.append(
                CUB_Loss(
                    loss.item(),
                    reconstruction_loss.item(),
                    representation_loss.item(),
                )
            )

            average_loss = _get_average_loss_cub(losses)

        epoch_results.append(
            EpochResult(
                tag=experiment.tag,
                epoch=epoch_ndx,
                total_loss=average_loss.total_loss,
                reconstruction_loss=average_loss.reconstruction_loss,
                representation_loss=average_loss.representation_loss,
            )
        )

        bar.progress((epoch_ndx + 1) / experiment.num_epochs)

    return CUB_TrainResult(experiment.tag, models, epoch_results)


if __name__ == "__main__":
    basic_cub_exp = CUB_Experiment(tag="basic")
    _train(basic_cub_exp)
