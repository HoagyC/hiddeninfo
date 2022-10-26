"""
From yewsiang/ConceptBottleneck
"""

import pickle
from pathlib import Path
from typing import List

import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, BatchSampler


class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(
        self,
        pkl_file_paths,
        use_attr,
        no_img,
        uncertain_label,
        image_dir,
        n_class_attr,
        transform=None,
    ):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, "rb")))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data["img_path"]
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

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data["uncertain_attribute_label"]
            else:
                attr_label = img_data["attribute_label"]
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((N_ATTRIBUTES, self.n_class_attr))
                    one_hot_attr_label[np.arange(N_ATTRIBUTES), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, attr_label
        else:
            return img, class_label


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
    use_attr: bool,
    no_img,
    batch_size: int,
    uncertain_label: bool = False,
    n_class_attr: int = 2,
    image_dir: str = "images",
    resampling: bool = False,
    resol: int = 299,
):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
    """
    resized_resol = int(resol * 256 / 224)
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
                # transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ]
        )

    dataset = CUBDataset(
        pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform
    )
    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    if resampling:
        sampler = BatchSampler(
            ImbalancedDatasetSampler(dataset),
            batch_size=batch_size,
            drop_last=drop_last,
        )
        loader = DataLoader(dataset, batch_sampler=sampler)
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
    return loader


if __name__ == "__main__":
    train_data_path = "CUB_processed/train.pkl"
    data = load_data(
        pkl_paths=[train_data_path], use_attr=False, no_img=False, batch_size=100
    )