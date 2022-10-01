from typing import List
import dataclasses
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import torch

VECTOR_SIZE = 20
LATENT_SIZE = 20
HIDDEN_SIZE = 20
NUM_BATCHES = 10_000
BATCH_SIZE = 32
REPRESENTATION_LOSS_COEFFICIENT = 8


@dataclasses.dataclass
class Experiment:
    has_representation_loss: float
    has_missing_knowledge: bool


@dataclasses.dataclass
class Result:
    step: int
    loss: float
    reconstruction_loss: float
    representation_loss: float
    # The reconstruction losses for the first & second halves of the vector.
    reconstruction_loss_p1: float
    reconstruction_loss_p2: float


def main():
    results_baseline = _train(
        Experiment(
            has_representation_loss=False,
            has_missing_knowledge=False,
        )
    )
    results_representation_loss = _train(
        Experiment(
            has_representation_loss=True,
            has_missing_knowledge=False,
        )
    )
    results_missing_knowledge = _train(
        Experiment(
            has_representation_loss=False,
            has_missing_knowledge=True,
        )
    )

    df = pd.DataFrame(
        [
            *[
                dict(exp="baseline", **dataclasses.asdict(result))
                for result in results_baseline
            ],
            *[
                dict(exp="representation_loss", **dataclasses.asdict(result))
                for result in results_representation_loss
            ],
            *[
                dict(exp="results_missing_knowledge", **dataclasses.asdict(result))
                for result in results_missing_knowledge
            ],
        ]
    )
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
        sns.lineplot(data=df, x="step", y=loss_name, hue="exp", ax=ax)
    st.pyplot(fig)


@st.cache
def _train(experiment: Experiment) -> List[Result]:
    encoder = _create_encoder()
    decoder = _create_decoder()

    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()])
    reconstruction_loss_fn = torch.nn.MSELoss()
    representation_loss_fn = torch.nn.MSELoss()

    results = []
    for step in range(NUM_BATCHES + 1):
        optimizer.zero_grad()

        vector = torch.normal(mean=0, std=1, size=(BATCH_SIZE, VECTOR_SIZE))
        if experiment.has_missing_knowledge:
            replacement = torch.normal(
                mean=0, std=1, size=(BATCH_SIZE, VECTOR_SIZE - LATENT_SIZE)
            )
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
        else:
            representation_loss = torch.tensor(0)
        loss = (
            reconstruction_loss + representation_loss * REPRESENTATION_LOSS_COEFFICIENT
        )

        loss.backward()
        optimizer.step()

        reconstruction_loss_p1 = reconstruction_loss_fn(
            vector[:LATENT_SIZE], vector_reconstructed[:LATENT_SIZE]
        )
        reconstruction_loss_p2 = reconstruction_loss_fn(
            vector[LATENT_SIZE:], vector_reconstructed[LATENT_SIZE:]
        )

        if step % 100 == 0:
            results.append(Result(
                step=step,
                loss=loss.item(),
                reconstruction_loss=reconstruction_loss.item(),
                representation_loss=representation_loss.item(),
                reconstruction_loss_p1=reconstruction_loss_p1.item(),
                reconstruction_loss_p2=reconstruction_loss_p2.item(),
            ))
    return results


def _create_encoder() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(VECTOR_SIZE, HIDDEN_SIZE),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_SIZE, LATENT_SIZE),
    )


def _create_decoder() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(LATENT_SIZE, HIDDEN_SIZE),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_SIZE, VECTOR_SIZE),
    )


if __name__ == "__main__":
    main()
