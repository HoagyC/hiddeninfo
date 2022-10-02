from cmath import exp
from typing import List
from typing import Tuple
import dataclasses
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import pickle
import seaborn as sns
import streamlit as st
import torch

VECTOR_SIZE = 20
LATENT_SIZE = 10
HIDDEN_SIZE = 20
NUM_BATCHES = 100_000
BATCH_SIZE = 32
REPRESENTATION_LOSS_COEFFICIENT = 5
NUM_ITERATIONS = 3
VECTOR_P2_SCALE = 3

CACHE_DIR = pathlib.Path("out/cache")


@dataclasses.dataclass
class Experiment:
    tag: str
    has_representation_loss: float
    has_missing_knowledge: bool


@dataclasses.dataclass
class Model:
    tag: str
    iteration: int
    encoder: torch.nn.Module
    decoder: torch.nn.Module


@dataclasses.dataclass
class Result:
    tag: str
    iteration: int
    step: int
    loss: float
    reconstruction_loss: float
    representation_loss: float
    # The reconstruction losses for the first & second halves of the vector.
    reconstruction_loss_p1: float
    reconstruction_loss_p2: float


def main():
    st.header("Hidden info")

    if not CACHE_DIR.is_dir():
        CACHE_DIR.mkdir(parents=True)

    experiments = [
        Experiment(
            tag="baseline",
            has_representation_loss=False,
            has_missing_knowledge=False,
        ),
        Experiment(
            tag="representation_loss",
            has_representation_loss=True,
            has_missing_knowledge=False,
        ),
        Experiment(
            tag="missing_knowledge",
            has_representation_loss=False,
            has_missing_knowledge=True,
        ),
    ]

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

    df = pd.DataFrame([dataclasses.asdict(result) for result in results])
    st.write(df)

    losses = [
        "loss",
        "reconstruction_loss",
        "representation_loss",
        "reconstruction_loss_p1",
        "reconstruction_loss_p2",
    ]
    fig, axs = plt.subplots(1, len(losses), figsize=(5 * len(losses), 5))
    for loss_name, ax in zip(losses, axs):
        sns.lineplot(data=df, x="step", y=loss_name, hue="tag", ax=ax)
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


def _train(experiment: Experiment, iteration: int) -> Tuple[Model, List[Result]]:
    model = Model(
        tag=experiment.tag,
        iteration=iteration,
        encoder=_create_encoder(),
        decoder=_create_decoder(),
    )

    optimizer = torch.optim.Adam(
        [*model.encoder.parameters(), *model.decoder.parameters()]
    )
    reconstruction_loss_fn = torch.nn.MSELoss()
    representation_loss_fn = torch.nn.MSELoss()

    results = []
    for step in range(NUM_BATCHES + 1):
        optimizer.zero_grad()

        vector = _generate_vector_batch()
        if experiment.has_missing_knowledge:
            vector_input = torch.concat(
                [vector[:, :LATENT_SIZE], _generate_vector_batch()[:, LATENT_SIZE:]],
                dim=1,
            )
        else:
            vector_input = vector
        latent_repr = model.encoder(vector_input)
        vector_reconstructed = model.decoder(latent_repr)

        reconstruction_loss = reconstruction_loss_fn(vector, vector_reconstructed)
        if experiment.has_representation_loss:
            representation_loss = representation_loss_fn(
                vector[:, :LATENT_SIZE], latent_repr
            )
            reconstruction_loss_coefficient = 1 / (REPRESENTATION_LOSS_COEFFICIENT + 1)
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

        if step % 100 == 0:
            results.append(
                Result(
                    tag=experiment.tag,
                    iteration=iteration,
                    step=step,
                    loss=loss.item(),
                    reconstruction_loss=reconstruction_loss.item(),
                    representation_loss=representation_loss.item(),
                    reconstruction_loss_p1=reconstruction_loss_p1.item(),
                    reconstruction_loss_p2=reconstruction_loss_p2.item(),
                )
            )

    return model, results


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


if __name__ == "__main__":
    main()
