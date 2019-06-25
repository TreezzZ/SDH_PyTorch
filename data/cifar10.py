# -*- coding: utf-8 -*-

from data.transform import img_transform
from data.transform import Onehot

import numpy as np
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
from torch.utils.data.dataloader import DataLoader


import os
import sys
import pickle


def load_data_gist(root, num_query, num_train):
    """Load cifar10-gist data

    Parameters
        root: str
        Path of dataset

        num_query: int
        Number of query images

        num_train: int
        Number of training images

    Returns
        query_data: Tensor
        Query data

        query_targets: Tensor
        Query targets

        train_data: Tensor
        Training data

        train_targets: Tensor
        Training targets

        database_data: Tensor
        Database data

        database_targets: Tensor
        Database targets
    """
    # Load data
    mat_data = sio.loadmat(root)
    all_data = mat_data['traindata']
    all_data = np.vstack((all_data, mat_data['testdata']))
    all_targets = mat_data['traingnd'].astype(np.int)
    all_targets = np.concatenate((all_targets, mat_data['testgnd'].astype(np.int)))

    # Split data
    perm_index = np.random.permutation(all_data.shape[0])
    query_index = perm_index[:num_query]
    train_index = perm_index[num_query: num_query + num_train]

    query_data = all_data[query_index, :]
    query_targets = all_targets[query_index]

    train_data = all_data[train_index, :]
    train_targets = all_targets[train_index]

    database_data = all_data
    database_targets = all_targets

    return query_data, query_targets, train_data, train_targets, database_data, database_targets


def load_data(root,
              num_query,
              num_train,
              batch_size,
              num_workers,
              ):
    """Load cifar10 data

    Parameters
        root: str
        path of dataset

        num_query: int
        Number of query images

        num_train: int
        Number of training images

        batch_size: int
        Batch size

        num_workers: int
        Number of load workers

    Returns
        query_dataloader, train_dataloader, database_dataloader: DataLoader
    """
    CIFAR10.init(root, num_query, num_train)
    query_dataset = CIFAR10('query', transform=img_transform(), target_transform=Onehot())
    train_dataset = CIFAR10('train', transform=img_transform(), target_transform=Onehot())
    database_dataset = CIFAR10('database', transform=img_transform(), target_transform=Onehot())

    query_dataloader = DataLoader(query_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  )
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  )
    database_dataloader = DataLoader(database_dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     )

    return query_dataloader, train_dataloader, database_dataloader


class CIFAR10(data.Dataset):
    """CIFAR-10 dataset"""
    @staticmethod
    def init(root, num_query, num_train):
        data_list = ['data_batch_1',
                     'data_batch_2',
                     'data_batch_3',
                     'data_batch_4',
                     'data_batch_5',
                     'test_batch',
                     ]
        base_folder = 'cifar-10-batches-py'

        data = []
        targets = []

        for file_name in data_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        targets = np.array(targets)

        CIFAR10.ALL_IMG = data
        CIFAR10.ALL_TARGETS = targets

        # sort by class
        sort_index = CIFAR10.ALL_TARGETS.argsort()
        CIFAR10.ALL_IMG = CIFAR10.ALL_IMG[sort_index, :]
        CIFAR10.ALL_TARGETS = CIFAR10.ALL_TARGETS[sort_index]

        # (num_query / number of class) query images per class
        # (num_train / number of class) train images per class
        query_per_class = num_query // 10
        train_per_class = num_train // 10

        # permutate index (range 0 - 6000 per class)
        perm_index = np.random.permutation(CIFAR10.ALL_IMG.shape[0] // 10)
        query_index = perm_index[:query_per_class]
        train_index = perm_index[query_per_class: query_per_class + train_per_class]

        query_index = np.tile(query_index, 10)
        train_index = np.tile(train_index, 10)
        inc_index = np.array([i * (CIFAR10.ALL_IMG.shape[0] // 10) for i in range(10)])
        query_index = query_index + inc_index.repeat(query_per_class)
        train_index = train_index + inc_index.repeat(train_per_class)

        # split data, tags
        CIFAR10.QUERY_IMG = CIFAR10.ALL_IMG[query_index, :]
        CIFAR10.QUERY_TARGETS = CIFAR10.ALL_TARGETS[query_index]
        CIFAR10.TRAIN_IMG = CIFAR10.ALL_IMG[train_index, :]
        CIFAR10.TRAIN_TARGETS = CIFAR10.ALL_TARGETS[train_index]

    def __init__(self, mode='train',
                 transform=None, target_transform=None,
                 ):
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'train':
            self.img = CIFAR10.TRAIN_IMG
            self.targets = CIFAR10.TRAIN_TARGETS
        elif mode == 'query':
            self.img = CIFAR10.QUERY_IMG
            self.targets = CIFAR10.QUERY_TARGETS
        else:
            self.img = CIFAR10.ALL_IMG
            self.targets = CIFAR10.ALL_TARGETS

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.img[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.img)
