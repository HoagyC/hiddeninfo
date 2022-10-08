from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable
import copy
import dataclasses
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import streamlit as st
import torch

import experiments as exps
from experiments import Experiment


RESULTS_DIR = Path("out/results")
DATETIME_FMT = "%Y%m%d-%H%M%S"


@dataclasses.dataclass
class Model:
    encoder: torch.nn.Module
    decoder: torch.nn.Module


@dataclasses.dataclass
class Loss:
    total_loss: float
    reconstruction_loss: float
    representation_loss: float
    # The reconstruction losses for the first & second halves of the vector.
    reconstruction_loss_p1: float
    reconstruction_loss_p2: float


@dataclasses.dataclass
class StepResult:
    tag: str
    step: int
    total_loss: float
    reconstruction_loss: float
    representation_loss: float
    # The reconstruction losses for the first & second halves of the vector.
    reconstruction_loss_p1: float
    reconstruction_loss_p2: float


@dataclasses.dataclass
class TrainResult:
    tag: str
    models: List[Model]
    step_results: List[StepResult]
    # validation_result: StepResult


def main():
    st.header("Hidden info")

    original_experiments: List[Experiment] = exps.new_decoders
    retrain_models = st.checkbox("Retrain models", value=False)

    for n_models in [2, 4, 8, 16]:
        st.subheader(f"Training with {n_models} models")
        experiments = copy.deepcopy(original_experiments)
        for exp in experiments:
            suffix = f"_{n_models}models"
            exp.n_models = n_models
            exp.tag += suffix
            if exp.load_decoders_from_tag is not None:
                exp.load_decoders_from_tag += suffix
            if exp.load_encoders_from_tag is not None:
                exp.load_encoders_from_tag += suffix

        bar = st.progress(0.0)
        train_results: List[TrainResult] = []
        for i, experiment in enumerate(experiments):
            if retrain_models or _get_train_result_path(experiment.tag) is None:
                train_result = _train(experiment=experiment)
                _save_train_result(train_result)
            else:
                train_result = _load_train_result(experiment.tag)
            train_results.append(train_result)
            bar.progress((i + 1) / len(experiments))

        tags = [experiment.tag for experiment in experiments]
        df = pd.DataFrame(
            [
                dataclasses.asdict(result)
                for train_result in train_results
                for result in train_result.step_results
                if result.tag in tags
            ]
        )
        st.write(df)

        print("written dfs")
        losses = [
            "total_loss",
            "reconstruction_loss",
            "representation_loss",
            "reconstruction_loss_p1",
            "reconstruction_loss_p2",
        ]
        print(len(losses))
        fig, axs = plt.subplots(1, len(losses), figsize=(5 * len(losses), 5))

        # With binary data and zero info, ideal prediction is always 0.5
        ZERO_INFO_LOSS = 0.5**2

        for loss_name, ax in zip(losses, axs):
            sns.lineplot(data=df, x="step", y=loss_name, hue="tag", ax=ax)
            if loss_name == "reconstruction_loss_p2":
                sns.lineplot(
                    x=[0, max(df.step)],
                    y=[ZERO_INFO_LOSS, ZERO_INFO_LOSS],
                    label="zero info loss",
                    linestyle="dashed",
                    ax=ax,
                )
            ax.set_title(loss_name)
            ax.set_yscale("linear")
            ax.set_ylim(([0, 0.3]))

        fig.tight_layout()
        st.pyplot(fig)

        df = df[df["step"] == df["step"].max()]
        df = df.drop("step", axis=1)
        df = pd.melt(
            df,
            id_vars=["tag"],
            var_name="loss_type",
            value_name="loss_value",
        )
        grid = sns.FacetGrid(df, col="loss_type", sharex=False)
        grid.map(sns.barplot, "loss_value", "tag")
        st.pyplot(grid.fig)


### List of interventions
# Freezing
# Training end-to-end
# Dropout
#

## Things to vary
# Dimensions of hidden variation
# Number of models used in multi-model scenarios


def _train(experiment: Experiment) -> TrainResult:
    if experiment.load_decoders_from_tag is not None:
        decoder_train_result = _load_train_result(experiment.load_decoders_from_tag)
        assert len(decoder_train_result.models) == experiment.n_models
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
        assert len(encoder_train_result.models) == experiment.n_models
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

    optimizer = torch.optim.Adam(list(itertools.chain.from_iterable(all_params)))

    reconstruction_loss_fn: Callable  # I thought this was PYTHON
    if experiment.use_class:
        reconstruction_loss_fn = torch.nn.CrossEntropyLoss()
    else:
        reconstruction_loss_fn = torch.nn.MSELoss()

    representation_loss_fn = torch.nn.MSELoss()

    step_results = []
    for step in range(experiment.num_batches + 1):
        losses = []
        if experiment.end_to_end:
            model_perm = torch.randperm(len(models))

        for model_ndx in range(len(models)):
            encoder = models[model_ndx].encoder
            if experiment.end_to_end:
                decoder = models[model_perm[model_ndx]].decoder
            else:
                decoder = models[model_ndx].decoder

            optimizer.zero_grad()

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
            vector_reconstructed = decoder(latent_repr)
            if experiment.use_class:
                vector_reconstructed = vector_reconstructed.reshape(
                    experiment.batch_size, 2, experiment.vector_size
                )
                vector_target = vector.to(dtype=torch.long)
            else:
                vector_target = vector

            reconstruction_loss = reconstruction_loss_fn(
                vector_reconstructed, vector_target
            )
            if experiment.has_representation_loss:
                representation_loss = representation_loss_fn(
                    vector[:, : experiment.preferred_rep_size],
                    latent_repr[:, : experiment.preferred_rep_size],
                )
                reconstruction_loss_coefficient = 1 / (experiment.repr_loss_coef + 1)
                representation_loss_coefficient = experiment.repr_loss_coef / (
                    experiment.repr_loss_coef + 1
                )
            else:
                representation_loss = torch.tensor(0)
                reconstruction_loss_coefficient = 1
                representation_loss_coefficient = 0
            loss = (
                reconstruction_loss * reconstruction_loss_coefficient
                + representation_loss * representation_loss_coefficient
            )
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
            losses.append(
                Loss(
                    loss.item(),
                    reconstruction_loss.item(),
                    representation_loss.item(),
                    reconstruction_loss_p1.item(),
                    reconstruction_loss_p2.item(),
                )
            )
            # import pdb

            # pdb.set_trace()

        average_loss = _get_average_loss(losses)

        if step % 100 == 0:
            step_results.append(
                StepResult(
                    tag=experiment.tag,
                    step=step,
                    total_loss=average_loss.total_loss,
                    reconstruction_loss=average_loss.reconstruction_loss,
                    representation_loss=average_loss.representation_loss,
                    reconstruction_loss_p1=average_loss.reconstruction_loss_p1,
                    reconstruction_loss_p2=average_loss.reconstruction_loss_p2,
                )
            )
        if step % 1000 == 0:
            print(vector[0], vector_input[0], latent_repr[0], vector_reconstructed[0])
            print(step)

    return TrainResult(experiment.tag, models, step_results)


def _get_average_loss(losses: List[Loss]) -> Loss:
    if len(losses) == 1:
        return losses[0]
    else:
        return Loss(
            float(np.mean([l.total_loss for l in losses])),
            float(np.mean([l.reconstruction_loss for l in losses])),
            float(np.mean([l.representation_loss for l in losses])),
            float(np.mean([l.reconstruction_loss_p1 for l in losses])),
            float(np.mean([l.reconstruction_loss_p2 for l in losses])),
        )


def _get_final_result(results: List[StepResult]) -> float:
    assert len(results) >= 200
    if len(results) >= 2000:
        mean_p2_loss = np.mean([r.reconstruction_loss_p2 for r in results[-1000:]])
    else:
        mean_p2_loss = np.mean([r.reconstruction_loss_p2 for r in results[-100:]])

    return float(mean_p2_loss)


def _create_encoder(
    vector_size: int,
    hidden_size: int,
    latent_size: int,
    n_hidden_layers: int,
    activation_fn: torch.nn.Module,
) -> torch.nn.Module:
    # TODO: Use layers.append here, like _create_decoder.
    in_layer = torch.nn.Linear(vector_size, hidden_size)
    hidden_layers = [
        torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            activation_fn,
        )
        for _ in range(n_hidden_layers)
    ]
    hidden_layers_seq = torch.nn.Sequential(*hidden_layers)
    out_layer = torch.nn.Linear(hidden_size, latent_size)

    return torch.nn.Sequential(in_layer, activation_fn, hidden_layers_seq, out_layer)


def _create_decoder(
    latent_size: int,
    hidden_size: int,
    vector_size: int,
    use_class: bool,
    n_hidden_layers: int,
    activation_fn: torch.nn.Module,
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
    layers.extend(
        torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), activation_fn)
        for _ in range(n_hidden_layers)
    )
    layers.append(torch.nn.Linear(hidden_size, output_size))
    return torch.nn.Sequential(*layers)


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


def _save_train_result(train_result: TrainResult):
    now_str = datetime.now().strftime(DATETIME_FMT)
    train_result_path = RESULTS_DIR / train_result.tag / now_str / "train-result.pickle"
    if not train_result_path.parent.is_dir():
        train_result_path.parent.mkdir(parents=True)
    with train_result_path.open("wb") as f:
        pickle.dump(train_result, f)


def _load_train_result(tag: str) -> TrainResult:
    train_result_path = _get_train_result_path(tag)
    assert train_result_path is not None, f"No train results for {tag}"
    with train_result_path.open("rb") as f:
        return pickle.load(f)


def _get_train_result_path(tag: str) -> Optional[Path]:
    results_dir = RESULTS_DIR / tag
    if not results_dir.is_dir():
        return None
    time_strs = [time_dir.name for time_dir in results_dir.iterdir()]
    times = [datetime.strptime(time_str, DATETIME_FMT) for time_str in time_strs]
    latest_time_str = max(times).strftime(DATETIME_FMT)
    return RESULTS_DIR / tag / latest_time_str / "train-result.pickle"


if __name__ == "__main__":
    main()
