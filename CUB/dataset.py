"""
From yewsiang/ConceptBottleneck
"""

import os
import pickle
from pathlib import Path
from typing import List
from datetime import datetime

from PIL import Image

import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, BatchSampler
from CUB.cub_classes import BASE_DIR, N_CLASSES, N_ATTRIBUTES


class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(
        self,
        pkl_file_paths,
        uncertain_label,
        image_dir,
        transform=None,
        example_attr_sparsity: int = 1,
        attr_class_sparsity: int = 1,
    ):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, "rb")))
        self.transform = transform
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.example_attr_sparsity = example_attr_sparsity


        # Randomly decide which classes to mask, note that mask=1 means visible
        class_mask = torch.ones(N_CLASSES).to(torch.bool)
        n_class_visible = N_CLASSES // attr_class_sparsity
        class_mask[-n_class_visible:] = 1
        self.class_mask = class_mask[torch.randperm(N_CLASSES)]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_data = self.data[idx]
        img_path = img_data["img_path"]
        attr_mask_bin = idx % self.example_attr_sparsity == 0
        class_mask_bin = self.class_mask[img_data["class_label"]]

        mask = attr_mask_bin and class_mask_bin

        # Trim unnecessary paths
        try:
            idx = img_path.split("/").index("CUB_200_2011")
            if self.image_dir != "images":
                img_path = "/".join([self.image_dir] + img_path.split("/")[idx + 1 :])
                img_path = img_path.replace("images/", "")
            else:
                img_path = "/".join(img_path.split("/")[idx:])
            img = Image.open(img_path).convert("RGB")
        except:
            img_path_split = img_path.split("/")
            split = "train" if self.is_train else "test"
            img_path = "/".join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert("RGB")
        
        class_label = img_data["class_label"]
        if self.transform:
            img = self.transform(img)

        if self.uncertain_label:
            attr_label = img_data["uncertain_attribute_label"]
        else:
            attr_label = img_data["attribute_label"]

        return img, class_label, attr_label, mask



class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [
            1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices
        ]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]["attribute_label"][0]

    def __iter__(self):
        idx = (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )
        return idx

    def __len__(self):
        return self.num_samples


def load_data(
    pkl_paths: List[str],
    args,
    resol: int = 299,
    uncertain_label: bool = False,
) -> DataLoader:
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
    """

    is_training = any(["train.pkl" in f for f in pkl_paths])
    if is_training:
        transform = transforms.Compose(
            [
                # transforms.Resize((resized_resol, resized_resol)),
                # transforms.RandomSizedCrop(resol),
                transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
                # transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                # transforms.Resize((resized_resol, resized_resol)),
                transforms.CenterCrop(resol),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
            ]
        )
    # Some configs were created before class_sparsity was added
    if hasattr(args, "class_sparsity"):
        class_sparsity = args.class_sparsity
    else:
        class_sparsity = 1

    dataset = CUBDataset(
        pkl_paths,
        uncertain_label,
        args.image_dir,
        transform,
        args.attr_sparsity,
        class_sparsity,
    )
    if is_training:
        drop_last = True # drop last batch if it is smaller than batch_size
        shuffle = True
    else:
        drop_last = False
        shuffle = False
        
    print(f"making dataloader where shuffle={shuffle} and drop_last={drop_last}")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=1
    )
    return loader


def find_class_imbalance(pkl_file, multiple_attr=False, attr_idx=-1):
    """
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    """
    imbalance_ratio = []
    data = pickle.load(open(os.path.join(BASE_DIR, pkl_file), "rb"))
    n = len(data)
    n_attr = len(data[0]["attribute_label"])
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for d in data:
        labels = d["attribute_label"]
        if multiple_attr:
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += labels[attr_idx]
            else:
                n_ones[0] += sum(labels)
    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j] / n_ones[j] - 1)
    if not multiple_attr:  # e.g. [9.0] --> [9.0] * 312
        imbalance_ratio *= n_attr
    return imbalance_ratio
