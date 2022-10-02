from cmath import exp
import itertools
from typing import List
import dataclasses
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import pathlib
import pickle

VECTOR_SIZE = 20
LATENT_SIZE = 10
HIDDEN_SIZE = 20
NUM_BATCHES = 10_000
BATCH_SIZE = 32
REPRESENTATION_LOSS_COEFFICIENT = 5
NUM_ITERATIONS = 3

CACHE_DIR = pathlib.Path("out/cache")


@dataclasses.dataclass
class Experiment:
    tag: str
    has_representation_loss: float
    has_missing_knowledge: bool


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

    if st.checkbox("Refresh results", value=False):
        results = []
        iterations = itertools.product(experiments, range(NUM_ITERATIONS))
        bar = st.progress(0.0)
        for i, (experiment, iteration) in enumerate(iterations):
            results.extend(_train(experiment, iteration))
            bar.progress(i / (len(experiments) * NUM_ITERATIONS))
        bar.progress(1.0)
        with (CACHE_DIR / "results.pickle").open("wb") as f:
            pickle.dump(results, f)
    else:
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



def _train(experiment: Experiment, iteration: int) -> List[Result]:
    encoder = _create_encoder()
    decoder = _create_decoder()

    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()])
    reconstruction_loss_fn = torch.nn.MSELoss()
    representation_loss_fn = torch.nn.MSELoss()

    results = []
    for step in range(NUM_BATCHES + 1):
        optimizer.zero_grad()

        vector = torch.concat(
            [
                _generate_vector_batch(LATENT_SIZE),
                _generate_vector_batch(VECTOR_SIZE - LATENT_SIZE) * 3,
            ],
            dim=1,
        )
        if experiment.has_missing_knowledge:
            replacement = _generate_vector_batch(VECTOR_SIZE - LATENT_SIZE) * 3
            vector_input = torch.concat([vector[:, :LATENT_SIZE], replacement], dim=1)
        else:
            vector_input = vector
        latent_repr = encoder(vector_input)
        vector_reconstructed = decoder(latent_repr)

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
    return results


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


def _generate_vector_batch(vector_size: int = VECTOR_SIZE) -> torch.Tensor:
    # High is exclusive, so add one.
    return torch.randint(low=0, high=2, size=(BATCH_SIZE, vector_size)).float()
    # return torch.normal(mean=0, std=1, size=(BATCH_SIZE, vector_size))


if __name__ == "__main__":
    main()
