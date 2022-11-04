import dataclasses
import os 
from pathlib import Path
import pickle
from typing import Dict, List

import torch

from torch.utils.data import Dataset, DataLoader

CLIP_DIM = 512
NUM_CLASSES = 200
HIDDEN_DIM = 1024
N_EPOCHS = 500

DATA_ADDR = Path("../out/CLIP_embeddings.pkl")
CUB_PATH = Path("../CUB_200_2011")

class CLIP_CUB_entry:
    ndx: int
    CLIP_embedding: torch.Tensor
    entry_class: int
    is_train: bool
    image_path: str


class CLIP_CUB_Dataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx: int) -> int:
        entry = self.data[ndx]
        return entry.CLIP_embedding, entry.entry_class


def _make_data_dict() -> Dict:
    path_to_id_dict = dict() #map from full image path to image id
    id_to_path_dict = dict()
    with open(CUB_PATH / "images.txt" 'r') as f:
        for line in f:
            items = line.strip().split()
            path = join(data_path, items[1])
            ndx = int(items[0])
            path_to_id_dict[path] = ndx
            id_to_path_dict[ndx] = path
            
    is_train_test = dict() #map from image id to 0 / 1 (1 = train)
    with open(CUB_PATH + 'train_test_split.txt', 'r') as f:
        for line in f:
            idx, is_train = line.strip().split()
            is_train_test[int(idx)] = int(is_train)

    id_to_class_dict = dict()
    with open(CUB_PATH + '/image_class_labels.txt', 'r') as f:
        for line in f:
            idx, class_label = line.strip().split()
            is_train_test[int(idx)] = int(class_label)
            
            
    with open(DATA_ADDR, "rb") as f:
        CLIP_data = pickle.load(f)
 
 
    data = []
    for ndx in id_to_class_dict.keys():
        item_path = id_to_path_dict[ndx]
        item_embedding = CLIP_data[item_path]
        is_train = bool(is_train_test[ndx])
        item_class = id_to_class_dict[ndx]
        data_entry = CLIP_CUB_entry(
            ndx = ndx,
            CLIP_embedding=item_embedding,
            is_train=is_train,
            entry_class=item_class,
            image_path=item_path,
        )
        data.append = data_entry

    return data
    
    
# def _train(dataloader):
#     f

def main():
    model = torch.nn.Sequential(
        torch.nn.Linear(CLIP_DIM, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, NUM_CLASSES),
    )
    
    data = _make_data_dict()
    
    train_data = [d for d in data if d.is_train]
    test_data = [d for d in data if not d.is_train]     
    
    train_dataset = CLIP_CUB_Dataset(train_data)
    test_dataset = CLIP_CUB_Dataset(test_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size = 32)
    test_dataloader = DataLoader(test_dataset, batch_size = 32)

    for data_batch in train_dataloader:
        print(data_batch)

if __name__ == "__main__":
    main()
