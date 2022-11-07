from typing import List
import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch

VECTOR_SIZE = 5
HIDDEN_SIZE = 16
LATENT_SIZE = 5
N_HIDDEN_LAYERS = 0
NUM_BATCHES = 20_000
NUM_BATCHES_PER_PHASE = 1_000
ADVERSARIAL_LOSS_COEFFICIENT = 1e-2


@dataclasses.dataclass
class Result:
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    latent_predictor: torch.nn.Module
    df: pd.DataFrame


def main():
    st.title("Adversarial embeddings")
    st.write(
        """
        This experiment tries to create latent spaces where each variable is independent over every
        other variable. This is done via an adversarial setup:
        - An **autoencoder** is trained to reconstruct a binary string.
        - A **latent predictor** is trained to predict latent variable $l_i$ given all other latent
          variables $l_j, i \\neq j$.
        - The autoencoder is has an additional cost term, which is the negative of the latent
          predictor's cost function.
        - We alternate between training the autoencoder and the latent predictor.
        """
    )
    if not st.checkbox("Run?"):
        return

    use_adversarial_loss = st.checkbox("Use adversarial loss?", True)

    st.header("Training")
    result = _train(use_adversarial_loss)

    st.header("Results")
    st.write(
        """
        Loss curves for:
        1. The reconstruction loss of the autoencoder.
        2. The prediction loss for the latent predictor.
        3. The total loss for the autoencoder.
        """
    )
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    sns.lineplot(data=result.df, x="step", y="reconstruction_loss", ax=ax1)
    sns.lineplot(data=result.df, x="step", y="latent_predictor_loss", ax=ax2)
    sns.lineplot(data=result.df, x="step", y="autoencoder_loss", ax=ax3)
    st.pyplot(fig)

    with torch.no_grad():
        inputs = torch.randint(low=0, high=2, size=(320, VECTOR_SIZE)).float()
        latents = result.encoder(inputs)
        latent_predictions = result.latent_predictor(latents)
        latent_predictor_loss = torch.mean((latents - latent_predictions) ** 2, dim=0)

    st.subheader("Correlations between latent variables")
    st.write(
        """
        We want this to be the identity matrix: Each latent variable is uncorrelated from every
        other latent variable.
        """
    )
    fig, ax = plt.subplots()
    sns.heatmap(
        torch.corrcoef(latents.T).numpy(), vmin=-1, vmax=1, center=0, annot=True, ax=ax
    )
    st.pyplot(fig)

    st.subheader("Latent predictor losses")
    st.write(
        """
        This is given per-latent variable. We want this to be ~0.25 (range [0, 1], average error of
        0.5, average squared error of 0.25). We also expect if there is a correlation between two
        latent variables above, this is reflected in a lower loss here.
        """
    )
    fig, ax = plt.subplots()
    ax.bar(height=latent_predictor_loss, x=list(range(LATENT_SIZE)))
    st.pyplot(fig)

    st.subheader("Inspecting latent variables")
    st.write(
        """
        We hypothesis that with N variables to encode in N *independent* latent variables, each
        latent variable will represent one-and-only-one variable. We can check the correspondences
        by looking at the correlations between the variables & latent variables.
        """
    )
    fig, ax = plt.subplots()
    inputs_latents_corr = np.array(
        [
            [np.corrcoef(inputs[:, i], latents[:, j])[0, 1] for i in range(LATENT_SIZE)]
            for j in range(LATENT_SIZE)
        ]
    )
    sns.heatmap(inputs_latents_corr, vmin=-1, vmax=1, center=0, annot=True, ax=ax)
    st.pyplot(fig)


@st.cache(suppress_st_warning=True)
def _train(use_adversarial_loss: bool) -> Result:
    encoder = _create_encoder(
        vector_size=VECTOR_SIZE,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        n_hidden_layers=N_HIDDEN_LAYERS,
    )
    decoder = _create_decoder(
        vector_size=VECTOR_SIZE,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        n_hidden_layers=N_HIDDEN_LAYERS,
    )
    latent_predictor = LatentPredictor(latent_size=LATENT_SIZE)

    autoencoder_optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters()]
    )
    latent_predictor_optimizer = torch.optim.Adam(latent_predictor.parameters())
    cost_fn = torch.nn.MSELoss()

    df = []
    bar = st.progress(0.0)
    for step in range(NUM_BATCHES):
        autoencoder_optimizer.zero_grad()
        latent_predictor_optimizer.zero_grad()
        inputs = torch.randint(low=0, high=2, size=(32, VECTOR_SIZE)).float()
        latents = encoder(inputs)
        outputs = decoder(latents)
        latent_predictions = latent_predictor(latents)

        reconstruction_loss = cost_fn(inputs, outputs)
        latent_predictor_loss = cost_fn(latent_predictions, latents)
        autoencoder_loss = reconstruction_loss - latent_predictor_loss * (
            ADVERSARIAL_LOSS_COEFFICIENT if use_adversarial_loss else 0
        )

        if (step // NUM_BATCHES_PER_PHASE) % 2 == 0:
            autoencoder_loss.backward()
            autoencoder_optimizer.step()
        else:
            latent_predictor_loss.backward()
            latent_predictor_optimizer.step()

        if step % 200 == 0:
            df.append(
                dict(
                    step=step,
                    reconstruction_loss=reconstruction_loss.item(),
                    latent_predictor_loss=latent_predictor_loss.item(),
                    autoencoder_loss=autoencoder_loss.item(),
                )
            )
        bar.progress((step + 1) / NUM_BATCHES)
    return Result(encoder, decoder, latent_predictor, pd.DataFrame(df))


def _create_encoder(
    vector_size: int,
    latent_size: int,
    hidden_size: int,
    n_hidden_layers: int,
) -> torch.nn.Module:
    layers: List[torch.nn.Module] = []
    layers.append(torch.nn.Linear(vector_size, hidden_size))
    for _ in range(n_hidden_layers):
        layers.append(torch.nn.Linear(hidden_size, hidden_size))
        layers.append(torch.nn.Sigmoid())
    layers.append(torch.nn.Linear(hidden_size, latent_size))
    layers.append(torch.nn.Sigmoid())
    return torch.nn.Sequential(*layers)


def _create_decoder(
    vector_size: int,
    latent_size: int,
    hidden_size: int,
    n_hidden_layers: int,
) -> torch.nn.Module:
    layers: List[torch.nn.Module] = []
    layers.append(torch.nn.Linear(latent_size, hidden_size))
    for _ in range(n_hidden_layers):
        layers.append(torch.nn.Linear(hidden_size, hidden_size))
        layers.append(torch.nn.Sigmoid())
    layers.append(torch.nn.Linear(hidden_size, vector_size))
    layers.append(torch.nn.Sigmoid())
    return torch.nn.Sequential(*layers)


class LatentPredictor(torch.nn.Module):
    def __init__(self, latent_size: int) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.weights = torch.nn.Parameter(torch.randn(latent_size, latent_size))
        self.biases = torch.nn.Parameter(torch.randn(latent_size))
        self.mask = 1 - torch.eye(self.latent_size)

    def forward(self, x) -> torch.Tensor:
        return torch.sigmoid(x @ (self.weights * self.mask) + self.biases)


if __name__ == "__main__":
    main()
