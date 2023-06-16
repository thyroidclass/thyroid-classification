import numpy as np
import torch
import h5py

import csv
import random
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from math import ceil, floor, sqrt
from sklearn.model_selection import StratifiedKFold

from utils.augmentation import get_cyto_transform

class ImageSet(Dataset):
    def __init__(self, dataset, tvt, istest=False, transform=None):
        self.dataset = dataset
        self.base_dir = f"./dataset/{dataset}"
        self.tvt = tvt
        self.istest = istest
        self.transform = transform

        self.stride = 128
        self.padsize = 256
        self.randseed = 831

        self.info_items = []
        self.items = []
        self.list_up()

    def list_up(self):
        with open(f"{self.base_dir}/{self.tvt}.csv", 'r') as f:
            reader = csv.reader(f)
            self.stride, self.padsize = next(reader)
            for r in reader:
                path, i, j, label = r
                self.info_items.append([path, int(i), int(j), int(label)])
        if not self.istest:
            random.seed(self.randseed)
            random.shuffle(self.info_items)

        self.stride = int(self.stride)
        self.padsize = int(self.padsize)

    def load_dataset(self, idx):

        if self.dataset.count('BF'): 
            path, i, j, label = self.info_items[idx]
            with h5py.File(path, 'r') as f:
                clst = np.array(f['bf'], dtype='float32')
            patch = clst[:, i * self.stride : i * self.stride + self.padsize, j * self.stride : j * self.stride + self.padsize]
            if self.istest:
                return (path, i, j, patch, label)
            else:
                return (patch, label)

        elif self.dataset.count('MIP'):
            path, i, j, label = self.info_items[idx]
            with h5py.File(path, 'r') as f:
                clst = np.array(f['mip'], dtype='float32')
                clst = np.expand_dims(clst, 0)
            patch = clst[:, i * self.stride : i * self.stride + self.padsize, j * self.stride : j * self.stride + self.padsize]
            if self.istest:
                return (path, i, j, patch, label)
            else:
                return (patch, label)            

    def __getitem__(self, idx):
        if self.istest:
            path, i, j, patch, label = self.load_dataset(idx)
            if self.transform:
                patch = self.transform(np.uint8(patch.transpose(1,2,0)))

            return path, i, j, patch, label
        else:
            patch, label = self.load_dataset(idx)
            if self.transform:
                patch = self.transform(np.uint8(patch.transpose(1,2,0)))
            return patch, label

    def __len__(self):
        return len(self.info_items)

    def __call__(self):
        return np.array(self.info_items)[:, 3]

def _get_split_indices_cls(trainset, p, seed):
    indices = list(range(len(trainset)))
    return [indices]


def _get_split_indices_rgs(trainset, p, seed):
    length = len(trainset)
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    sep = int(length * p)
    return indices[sep:], indices[:sep]


def _get_kfolded_indices_rgs(valid_indices, trainset, num_k, seed):
    np.random.seed(seed)
    valid_indices = np.array(valid_indices)
    np.random.shuffle(valid_indices)
    if len(valid_indices) % num_k:
        valid_indices = np.pad(
            valid_indices, (0, num_k - len(valid_indices) % num_k), mode="edge"
        )
    valid_indices = valid_indices.reshape(num_k, -1)
    return valid_indices

class NbsDataset(ImageSet):
    def __init__(self, data_dir, dataset, group, transform):
        super().__init__(dataset=data_dir, tvt=dataset, transform=transform)
        self.group = group

    def __getitem__(self, idx):
        index = np.where(self.group == idx)[0][0]
        patch, label = self.loading_realtime(idx)

        if self.transform:
            patch = patch.transpose(2, 1, 0)
            patch = self.transform(np.uint8(patch))

        return patch, label, index

class BaseDataLoader(object):
    def __init__(self, dataset, batch_size, cpus, with_index, seed, val_splitter):
        self.with_index = with_index
        self.data_dir = dataset
        self.dataset = self._get_dataset(dataset)
        self.split_indices = val_splitter(self.dataset['train'], 0.1, seed)

        self.n_train = len(self.dataset['train'])
        self.n_val = len(self.dataset['val'])
        self.n_test = len(self.dataset['test'])
        self.batch_size = batch_size
        self.cpus = cpus

    def load(self, phase):
        _f = {
            "train": lambda: self._train(),
            "val": lambda: self._val(),
            "test": lambda: self._test(),
        }
        try:
            loader = _f[phase]()
            return loader
        except KeyError:
            raise ValueError("Dataset should be one of [train, val, test]")

    def _train(self,):
        self.len = ceil(self.n_train / self.batch_size)

        dataset = (
            NbsDataset(self.data_dir, "train", self.groups, self.dataset["train"].transform)
            if self.with_index
            else self.dataset["train"]
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cpus,
            pin_memory=True,
        )
        return loader

    def _val(self):
        self.len = ceil(self.n_val / self.batch_size)

        loader = DataLoader(
            self.dataset["val"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cpus,
            pin_memory=True,
        )
        return loader

    def _test(self):
        self.len = ceil(self.n_test / self.batch_size)
        loader = DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            num_workers=self.cpus,
            pin_memory=True,
        )
        return loader

    def _get_dataset(self, dataset):
        if self.data_dir.count('BF'):
            dtype = 'BF'
        elif self.data_dir.count('MIP'):
            dtype = 'MIP'
        else:
            dtype = 'none'

        _d = self._load_cytology(dataset, dtype)

        try:
            _dataset = _d
            return _dataset
        except KeyError:
            raise ValueError(
                "Unknown dataset."
            )

    def _load_cytology(self, dataset, dtype=None):

        train_transforms = get_cyto_transform(224, 64, 8, dtype)['train']
        test_transforms = get_cyto_transform(256, 64, 8, dtype)['test']

        trainset = ImageSet(dataset, 'train', transform=train_transforms)
        validset = ImageSet(dataset, 'val',  transform=test_transforms)
        testset = ImageSet(dataset, 'test', True, transform=test_transforms)

        return {"train": trainset, "val": validset, "test": testset}


class GeneralDataLoaderCls(BaseDataLoader):
    def __init__(
        self, dataset, is_cluster, aug, batch_size, cpus, seed=0, val_splitter=_get_split_indices_cls
    ):
        super().__init__(dataset, is_cluster, aug, batch_size, cpus, False, seed, val_splitter)


class NbsDataLoaderCls(BaseDataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        n_a,
        cpus,
        seed=0,
        val_splitter=_get_split_indices_cls,
    ):
        super().__init__(dataset, batch_size, cpus, True, seed, val_splitter)
        self.n_a = n_a
        self.groups = _get_kfolded_indices_rgs(
            self.split_indices[0], self.dataset["train"], n_a, seed
        )


class GeneralDataLoaderRgs(BaseDataLoader):
    def __init__(
        self, dataset, batch_size, cpus, seed=0, val_splitter=_get_split_indices_rgs
    ):
        super().__init__(dataset, batch_size, cpus, False, seed, val_splitter)


class NbsDataLoaderRgs(BaseDataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        n_a,
        cpus,
        seed=0,
        val_splitter=_get_split_indices_rgs,
    ):
        super().__init__(dataset, batch_size, cpus, True, seed, val_splitter)
        self.n_a = n_a
        self.groups = _get_kfolded_indices_rgs(
            self.split_indices[0], self.dataset["train"], n_a, seed
        )


class GeneralDataLoaderSeg(GeneralDataLoaderRgs):
    pass


class NbsDataLoaderSeg(NbsDataLoaderRgs):
    pass
