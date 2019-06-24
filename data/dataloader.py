#!/usr/bin/env python
# -*- coding: utf-8 -*-

import data.cifar10 as cifar10
import data.nus_wide as nus_wide


def load_data(data_path,
              dataset,
              num_query,
              num_train,
              batch_size,
              num_workers,
              ):
    """Load data

    Parameters
        data_path: str
        path of dataset

        dataset: str
        dataset used to process

        num_query: int
        Number of query images

        num_train: int
        Number of training images

        batch_size: int
        Batch size

        num_workers: int
        Number of load workers

    Returns
        DataLoader, for cifar10 and nus-wide; Tensor, for cifar10-gist
    """
    if dataset == 'cifar10':
        return cifar10.load_data(data_path,
                                 num_query,
                                 num_train,
                                 batch_size,
                                 num_workers,
                                 )
    elif dataset == 'nus-wide':
        return nus_wide.load_data(data_path,
                                  num_query,
                                  num_train,
                                  batch_size,
                                  num_workers,
                                  )
    elif dataset == 'cifar10-gist':
        return cifar10.load_data_gist(data_path,
                                      num_query,
                                      num_train,
                                      )
