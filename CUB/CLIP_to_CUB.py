import os 
from pathlib import Path
import pickle

import torch

from torch.utils.data import Dataset, DataLoader

CLIP_DIM = 512
NUM_CLASSES = 200
HIDDEN_DIM = 1024
N_EPOCHS = 500

DATA_ADDR = Path("../out/CLIP_embeddings.pkl")

def path_to_ndx:
    

class CLIP_CUB_Dataset(Dataset):
    def __init__(self, CLIP_dict):
        self.CLIP_dict = CLIP_dict

    def __len__(self):
        return len(self.clip_dict)

    def __getitem__(self, ndx: int) -> int:
        pass


def main():
    model = torch.nn.Sequential(
        torch.nn.Linear(CLIP_DIM, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, NUM_CLASSES),
    )

    with open(DATA_ADDR, "rb") as f:
        CLIP_data = pickle.load(f)


if __name__ == "__main__":
    main()
