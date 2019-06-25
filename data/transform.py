#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import numpy as np
import torchvision.transforms as transforms


class Onehot(object):
    """one-hot transform
    """

    def __init__(self):
        pass

    def __call__(self, sample, num_classes=10):
        target_onehot = torch.zeros(num_classes)
        target_onehot[sample] = 1

        return target_onehot


def encode_onehot(labels, num_classes=10):
    """one-hot labels

    Parameters
        labels: ndarray
        Label

        num_classes: int
        Number of classes

    Returns
        onehot_labels: ndarray
        Onehot label
    """
    onehot_labels = np.zeros((len(labels), num_classes))

    for i in range(len(labels)):
        onehot_labels[i, labels[i]] = 1

    return onehot_labels


def img_transform():
    """Return transform

    Returns:
        transform: transforms
        Image transform
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform


def normalization(data):
    """normalize data, (data - mean) / std

    Parameters
        data: ndarray
        Data

    Returns
        normalized_data: ndarray
        Normalized data
    """
    if data.dtype != np.float:
        data = data.astype(np.float)
    return (data - data.mean()) / data.std()