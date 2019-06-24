# -*- coding:utf-8 -*-

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from data.transform import img_transform

import numpy as np
import os
from PIL import Image


def load_data(root,
              num_query,
              num_train,
              batch_size,
              num_workers,
              ):
    """加载NUS-WIDE数据集

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
        数据加载器
    """
    NUS_WIDE.init(root, num_query, num_train)
    query_dataset = NUS_WIDE('query', transform=img_transform())
    train_dataset = NUS_WIDE('train', transform=img_transform())
    database_dataset = NUS_WIDE('all', transform=img_transform())

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


class NUS_WIDE(Dataset):
    @staticmethod
    def init(path, num_query, num_train):
        # load data, tags
        NUS_WIDE.ALL_IMG = np.load(os.path.join(path, 'nus-wide-21-img.npy'))
        NUS_WIDE.ALL_TARGETS = np.load(os.path.join(path, 'nus-wide-21-tag.npy')).astype(np.float32)
        NUS_WIDE.ALL_IMG = NUS_WIDE.ALL_IMG.transpose((0, 2, 3, 1))

        # split data, tags
        perm_index = np.random.permutation(NUS_WIDE.ALL_IMG.shape[0])
        query_index = perm_index[:num_query]
        train_index = perm_index[:num_train]

        NUS_WIDE.QUERY_IMG = NUS_WIDE.ALL_IMG[query_index, :]
        NUS_WIDE.QUERY_TARGETS = NUS_WIDE.ALL_TARGETS[query_index, :]
        NUS_WIDE.TRAIN_IMG = NUS_WIDE.ALL_IMG[train_index, :]
        NUS_WIDE.TRAIN_TARGETS = NUS_WIDE.ALL_TARGETS[train_index, :]

    def __init__(self, mode, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'train':
            self.img = NUS_WIDE.TRAIN_IMG
            self.targets = NUS_WIDE.TRAIN_TARGETS
        elif mode == 'query':
            self.img = NUS_WIDE.QUERY_IMG
            self.targets = NUS_WIDE.QUERY_TARGETS
        else:
            self.img = NUS_WIDE.ALL_IMG
            self.targets = NUS_WIDE.ALL_TARGETS

    def __getitem__(self, index):
        img, target = self.img[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return self.img.shape[0]
