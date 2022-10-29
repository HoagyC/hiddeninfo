from typing import List
import streamlit as st
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def main():
    st.header("Adversarial embeddings")
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

    vector_size = 5
    hidden_size = 16
    latent_size = 5
    n_hidden_layers = 1
    # num_batches = 10_000
    num_batches = 100_000

    encoder = _create_encoder(
        vector_size=vector_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        n_hidden_layers=n_hidden_layers,
    )
    decoder = _create_decoder(
        vector_size=vector_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        n_hidden_layers=n_hidden_layers,
    )
    latent_predictor = LatentPredictor(latent_size=latent_size)

    autoencoder_optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters()]
    )
    latent_predictor_optimizer = torch.optim.Adam(latent_predictor.parameters())
    cost_fn = torch.nn.MSELoss()

    df = []
    bar = st.progress(0.0)
    for step in range(num_batches):
        autoencoder_optimizer.zero_grad()
        latent_predictor_optimizer.zero_grad()
        inputs = torch.randint(low=0, high=2, size=(32, vector_size)).float()
        latents = encoder(inputs)
        outputs = decoder(latents)
        latent_predictions = latent_predictor(latents)

        reconstruction_loss = cost_fn(inputs, outputs)
        latent_predictor_loss = cost_fn(latent_predictions, latents)
        autoencoder_loss = reconstruction_loss - latent_predictor_loss * 1e-2

        if (step // 1_000) % 2 == 0:
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
        bar.progress((step + 1) / num_batches)
    df = pd.DataFrame(df)

    with torch.no_grad():
        inputs = torch.randint(low=0, high=2, size=(320, vector_size)).float()
        latents = encoder(inputs)
        latent_predictions = latent_predictor(latents)
        latent_predictor_loss = torch.mean((latents - latent_predictions) ** 2, dim=0)

    fig, ax = plt.subplots()
    sns.heatmap(
        torch.corrcoef(latents.T).numpy(),
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        ax=ax,
    )
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.bar(height=latent_predictor_loss, x=list(range(latent_size)))
    st.pyplot(fig)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    sns.lineplot(data=df, x="step", y="reconstruction_loss", ax=ax1)
    sns.lineplot(data=df, x="step", y="latent_predictor_loss", ax=ax2)
    sns.lineplot(data=df, x="step", y="autoencoder_loss", ax=ax3)
    st.pyplot(fig)


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
