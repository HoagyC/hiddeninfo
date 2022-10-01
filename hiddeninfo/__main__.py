from typing import Iterable
import dataclasses
import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns

VECTOR_SIZE = 20
LATENT_SIZE = 20
HIDDEN_SIZE = 20
NUM_BATCHES = 1000
BATCH_SIZE = 32


@dataclasses.dataclass
class EncoderDecoder:
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    has_representation_loss: float
    # has_missing_knowledge: bool


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
    results_with_representation_loss = _train(
        EncoderDecoder(
            _create_encoder(),
            _create_decoder(),
            has_representation_loss=True,
        )
    )
    results_without_representation_loss = _train(
        EncoderDecoder(
            _create_encoder(),
            _create_decoder(),
            has_representation_loss=False,
        )
    )

    df = pd.DataFrame(
        [
            *[
                dict(exp="with_representation_loss", **dataclasses.asdict(result))
                for result in results_with_representation_loss
            ],
            * [
                dict(exp="without_representation_loss", **dataclasses.asdict(result))
                for result in results_without_representation_loss
            ]
        ]
    )
    sns.lineplot(data=df, x="step", y="loss", hue="exp")
    plt.show()


def _train(encoder_decoder: EncoderDecoder) -> Iterable[Result]:
    optimizer = torch.optim.Adam(
        [*encoder_decoder.encoder.parameters(), *encoder_decoder.decoder.parameters()]
    )
    reconstruction_loss_fn = torch.nn.MSELoss()
    representation_loss_fn = torch.nn.MSELoss()

    for step in range(NUM_BATCHES):
        optimizer.zero_grad()

        vector = torch.normal(mean=0, std=1, size=(BATCH_SIZE, VECTOR_SIZE))
        latent_repr = encoder_decoder.encoder(vector)
        vector_reconstructed = encoder_decoder.decoder(latent_repr)

        reconstruction_loss = reconstruction_loss_fn(vector, vector_reconstructed)
        if encoder_decoder.has_representation_loss:
            representation_loss = representation_loss_fn(
                vector[:, :LATENT_SIZE], latent_repr
            )
        else:
            representation_loss = torch.tensor(0)
        loss = reconstruction_loss + representation_loss

        loss.backward()
        optimizer.step()

        reconstruction_loss_p1 = reconstruction_loss_fn(
            vector[:LATENT_SIZE], vector_reconstructed[:LATENT_SIZE]
        )
        reconstruction_loss_p2 = reconstruction_loss_fn(
            vector[LATENT_SIZE:], vector_reconstructed[LATENT_SIZE:]
        )

        if step % 100 == 0:
            print(step, loss.item())
        yield Result(
            step=step,
            loss=loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            representation_loss=representation_loss.item(),
            reconstruction_loss_p1=reconstruction_loss_p1.item(),
            reconstruction_loss_p2=reconstruction_loss_p2.item(),
        )


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
