from codecs import latin_1_decode
import dataclasses
import torch

VECTOR_SIZE = 20
LATENT_SIZE = 20
HIDDEN_SIZE = 20
NUM_BATCHES = 10_000
BATCH_SIZE = 32


def main():
    encoder = _create_encoder(
        vector_size=VECTOR_SIZE, latent_size=LATENT_SIZE, hidden_size=HIDDEN_SIZE
    )
    decoder = _create_decoder(
        vector_size=VECTOR_SIZE, latent_size=LATENT_SIZE, hidden_size=HIDDEN_SIZE
    )

    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()])
    loss_fn = torch.nn.MSELoss()

    for step in range(NUM_BATCHES):
        optimizer.zero_grad()
        vector = torch.normal(mean=0, std=1, size=(BATCH_SIZE, VECTOR_SIZE))
        vector_reconstructed = decoder(encoder(vector))
        loss = loss_fn(vector, vector_reconstructed)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(step, loss.item())


def _create_encoder(
    *,
    vector_size: int,
    latent_size: int,
    hidden_size: int,
) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(vector_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, latent_size),
    )


def _create_decoder(
    *,
    vector_size: int,
    latent_size: int,
    hidden_size: int,
) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(latent_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, vector_size),
    )


if __name__ == "__main__":
    main()
