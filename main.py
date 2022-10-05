from typing import List
from typing import Tuple
import dataclasses
import functools
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sns
import streamlit as st
import torch

import experiments as exps
from experiments import Experiment

VECTOR_SIZE = 20
LATENT_SIZE = 10
HIDDEN_SIZE = 20
NUM_BATCHES = 5_000
BATCH_SIZE = 32
REPRESENTATION_LOSS_COEFFICIENT = 1
NUM_ITERATIONS = 1
VECTOR_P2_SCALE = 3
DROPOUT_P = 0.3

CACHE_DIR = Path("out/cache")


@dataclasses.dataclass
class Model:
    tag: str
    iteration: int
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
class Result:
    tag: str
    iteration: int
    step: int
    total_loss: float
    reconstruction_loss: float
    representation_loss: float
    # The reconstruction losses for the first & second halves of the vector.
    reconstruction_loss_p1: float
    reconstruction_loss_p2: float


def main():
    st.header("Hidden info")

    if not CACHE_DIR.is_dir():
        CACHE_DIR.mkdir(parents=True)

    experiments = [exps.prep_decoders3, exps.fresh_encoders3, exps.fresh_decoder]

    if st.checkbox("Retrain models", value=False):
        results = []
        models = []
        iterations = itertools.product(experiments, range(NUM_ITERATIONS))
        bar = st.progress(0.0)
        for i, (experiment, iteration) in enumerate(iterations):
            model, iteration_results = _train(experiment, iteration)
            models.append(model)
            results.extend(iteration_results)
            bar.progress(i / (len(experiments) * NUM_ITERATIONS))
        bar.progress(1.0)
        with (CACHE_DIR / "models.pickle").open("wb") as f:
            pickle.dump(models, f)
        with (CACHE_DIR / "results.pickle").open("wb") as f:
            pickle.dump(results, f)
    else:
        with (CACHE_DIR / "models.pickle").open("rb") as f:
            models = pickle.load(f)
        with (CACHE_DIR / "results.pickle").open("rb") as f:
            results = pickle.load(f)

        completed_exps = list(set([result.tag for result in results]))
        new_experiments = [exp for exp in experiments if exp.tag not in completed_exps]
        iterations = itertools.product(new_experiments, range(NUM_ITERATIONS))
        bar = st.progress(0.0)
        for i, (experiment, iteration) in enumerate(iterations):
            model, iteration_results = _train(experiment, iteration)
            models.append(model)
            results.extend(iteration_results)
            bar.progress(i / (len(new_experiments) * NUM_ITERATIONS))
        bar.progress(1.0)

    df = pd.DataFrame([dataclasses.asdict(result) for result in results])
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

    for loss_name, ax in zip(losses, axs):
        sns.lineplot(data=df, x="step", y=loss_name, hue="tag", ax=ax)
        print("linplot")
        ax.set_title(loss_name)
        ax.set_yscale("log")

    fig.tight_layout()
    st.pyplot(fig)

    df = df[df["step"] == df["step"].max()]
    df = df.drop("step", axis=1)
    df = pd.melt(
        df, id_vars=["tag", "iteration"], var_name="loss_type", value_name="loss_value"
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


def _train(experiment: Experiment, iteration: int) -> Tuple[List[Model], List[Result]]:
    if experiment.load_decoder:
        loaded_decoders = _load_decoders(experiment.decoder_loc)
        assert len(loaded_decoders) == experiment.n_models
        dec_fn = lambda x: loaded_decoders[x]
    else:
        dec_fn = lambda x: _create_decoder()

    if experiment.load_encoder:
        loaded_encoders = _load_encoders(experiment.encoder_loc)
        assert len(loaded_encoders) == experiment.n_models
        enc_fn = lambda x: loaded_encoders[x]
    else:
        enc_fn = lambda x: _create_encoder()

    models = [
        Model(
            tag=experiment.tag + str(ndx),
            iteration=iteration,
            encoder=enc_fn(ndx),
            decoder=dec_fn(ndx),
        )
        for ndx in range(experiment.n_models)
    ]

    if experiment.load_decoder:
        all_params = [[*model.encoder.parameters()] for model in models]
    elif experiment.load_encoder:
        all_params = [[*model.decoder.parameters()] for model in models]
    else:
        all_params = [
            [*model.encoder.parameters(), *model.decoder.parameters()]
            for model in models
        ]

    optimizer = torch.optim.Adam(list(itertools.chain.from_iterable(all_params)))
    reconstruction_loss_fn = torch.nn.MSELoss()
    representation_loss_fn = torch.nn.MSELoss()

    results = []
    for step in range(NUM_BATCHES + 1):
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

            vector = _generate_vector_batch()
            if experiment.has_missing_knowledge:
                vector_input = torch.concat(
                    [
                        vector[:, :LATENT_SIZE],
                        _generate_vector_batch()[:, LATENT_SIZE:],
                    ],
                    dim=1,
                )
            else:
                vector_input = vector

            latent_repr = encoder(vector_input)
            if experiment.dropout:
                # Applying dropout manually to avoid scaling
                bernoulli_t = torch.full(size=[HIDDEN_SIZE], fill_value=DROPOUT_P)
                dropout_t = torch.bernoulli(bernoulli_t)
                latent_repr = latent_repr * DROPOUT_P

            vector_reconstructed = decoder(latent_repr)

            reconstruction_loss = reconstruction_loss_fn(vector, vector_reconstructed)
            if experiment.has_representation_loss:
                representation_loss = representation_loss_fn(
                    vector[:, :LATENT_SIZE], latent_repr
                )
                reconstruction_loss_coefficient = 1 / (
                    REPRESENTATION_LOSS_COEFFICIENT + 1
                )
                representation_loss_coefficient = REPRESENTATION_LOSS_COEFFICIENT / (
                    REPRESENTATION_LOSS_COEFFICIENT + 1
                )
            else:
                representation_loss = torch.tensor(0)
                reconstruction_loss_coefficient = 1
                representation_loss_coefficient = 0
            loss = (
                reconstruction_loss * reconstruction_loss_coefficient
                + representation_loss * representation_loss_coefficient
            )

            loss.backward()
            optimizer.step()
            reconstruction_loss_p1 = reconstruction_loss_fn(
                vector[:, :LATENT_SIZE], vector_reconstructed[:, :LATENT_SIZE]
            )
            reconstruction_loss_p2 = reconstruction_loss_fn(
                vector[:, LATENT_SIZE:], vector_reconstructed[:, LATENT_SIZE:]
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

        average_loss = _get_average_loss(losses)

        if step % 100 == 0:
            results.append(
                Result(
                    tag=experiment.tag,
                    iteration=iteration,
                    step=step,
                    total_loss=average_loss.total_loss,
                    reconstruction_loss=average_loss.reconstruction_loss,
                    representation_loss=average_loss.representation_loss,
                    reconstruction_loss_p1=average_loss.reconstruction_loss_p1,
                    reconstruction_loss_p2=average_loss.reconstruction_loss_p2,
                )
            )
        if step % 10000 == 0:
            print(step)

    if experiment.save_model:
        _save_models(models=models, location=experiment.save_model)
    return models, results


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


def _create_encoder() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(VECTOR_SIZE, HIDDEN_SIZE),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_SIZE, LATENT_SIZE),
    )


def _create_decoder() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(LATENT_SIZE, HIDDEN_SIZE),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_SIZE, VECTOR_SIZE),
    )


def _generate_vector_batch() -> torch.Tensor:
    # High is exclusive, so add one.
    p1_high = 1 + 1
    p2_high = VECTOR_P2_SCALE + 1
    p1 = torch.randint(low=0, high=p1_high, size=(BATCH_SIZE, LATENT_SIZE)).float()
    p2 = torch.randint(
        low=0, high=p2_high, size=(BATCH_SIZE, VECTOR_SIZE - LATENT_SIZE)
    ).float()
    return torch.concat([p1, p2], dim=1)


def _load_decoders(location: Path) -> List[torch.nn.Module]:
    with open(location, "rb") as f:
        models = pickle.load(f)

    return [model.decoder for model in models]


def _load_encoders(location: Path) -> List[torch.nn.Module]:
    with open(location, "rb") as f:
        models = pickle.load(f)

    return [model.encoder for model in models]


def _save_models(models: List[Model], location: Path) -> None:
    if not location.parent.is_dir():
        location.parent.mkdir(parents=True)
    with open(location, "wb") as f:
        pickle.dump(models, f)


if __name__ == "__main__":
    main()
