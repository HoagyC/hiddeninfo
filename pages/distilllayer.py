import copy
import enum
import random
from typing import List
import torch


class DistillLayer(torch.nn.Module):
    def __init__(
        self,
        *,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        n: int,
        distillation_mode: "DistillationMode",
    ) -> None:
        self._base_encoder = encoder
        self._base_decoder = decoder
        self.n = n
        self.distillation_mode = distillation_mode
        # TODO: Should we re-init parameters?
        self._encoders = [copy.deepcopy(self._base_encoder) for _ in range(n)]
        self._decoders = [copy.deepcopy(self._base_decoder) for _ in range(n)]
        super().__init__()

    def forward(self, input: torch.Tensor, mode: "TrainingMode") -> List[torch.Tensor]:
        if mode == TrainingMode.SETUP:
            return self._forward_setup_mode(input)
        elif mode == TrainingMode.DISTILLATION:
            return self._forward_distill_mode(input)
        else:
            raise AssertionError()

    def _forward_setup_mode(self, input: torch.Tensor) -> List[torch.Tensor]:
        latents = [encoder(input) for encoder in self._encoders]
        return [decoder(latent) for latent, decoder in zip(latents, self._decoders)]

    def _forward_distill_mode(self, input: torch.Tensor) -> List[torch.Tensor]:
        encoders = random.sample(self._encoders)
        decoders = random.sample(self._decoders)
        if self.distillation_mode == DistillationMode.RETRAIN_DECODERS:
            with torch.no_grad():
                latents = [encoder(input) for encoder in encoders]
            return [decoder(latent) for latent, decoder in zip(latents, decoders)]
        return None


class DistillationMode(enum.Enum):
    RETRAIN_ENCODERS = 1
    RETRAIN_DECODERS = 2


class TrainingMode(enum.Enum):
    SETUP = 1
    DISTILLATION = 2


def main():
    l1 = torch.nn.Linear(10, 10)
    l2 = torch.nn.Linear(10, 10)
    l3 = torch.nn.Linear(10, 10)
    l2.weight.requires_grad = False
    l2.bias.requires_grad = False

    optim = torch.optim.Adam([*l1.parameters(), *l2.parameters(), *l3.parameters()])

    optim.zero_grad()
    x: torch.Tensor = torch.tensor([1] * 10).float()
    x = l1(x)
    x = l2(x)
    x = l3(x)
    x.sum().backward()
    optim.step()
    print(l1.weight.grad is not None)
    print(l2.weight.grad is not None)
    print(l3.weight.grad is not None)

    print("hello world")


if __name__ == "__main__":
    main()
