import torch
import numpy as np
import scipy.io as sio

from data.transform import encode_onehot


def load_data(dataset, root, num_train, num_query):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_train(int): Number of training data points.
        num_query(int): Number of query data points.

    Returns
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """
    if dataset == 'cifar-10-gist':
        train_data, train_targets, \
        query_data, query_targets, \
        retrieval_data, retrieval_targets = load_gist_data(root, num_train, num_query)

        # One-hot
        train_targets = torch.from_numpy(encode_onehot(train_targets, 10)).float()
        query_targets = torch.from_numpy(encode_onehot(query_targets, 10)).float()
        retrieval_targets = torch.from_numpy(encode_onehot(retrieval_targets, 10)).float()
    elif dataset == 'nus-wide-tc21' or dataset == 'imagenet-tc100':
        train_data, train_targets, \
        query_data, query_targets, \
        retrieval_data, retrieval_targets = _load_data(root, num_train, num_query)
    else:
        raise ValueError("Invalid dataset name!")

    return train_data, train_targets, query_data, query_targets, retrieval_data, retrieval_targets


def _load_data(root, num_train, num_query):
    """
    Load dataset.

    Args
        root(str): Path of dataset.
        num_train(int): Number of training data points.
        num_query(int): Number of query data points.

    Returns
        train_data(torch.Tensor): Training data points.
        train_targets(torch.Tensor): Training targets.
        query_data(torch.Tensor): Query data points.
        query_targets(torch.Tensor): Query targets.
        retrieval_data(torch.Tensor): Retrieval data points.
        retrieval_targets(torch.Tensor): Retrieval targets.
    """
    # Load data
    checkpoint = torch.load(root)
    data = checkpoint['features']
    targets = checkpoint['targets']

    # Normalization
    data = (data - data.mean()) / data.std()

    # Split
    n = data.shape[0]
    perm_index = torch.randperm(n)
    query_index = perm_index[:num_query]
    retrieval_index = perm_index[num_query:]
    train_index = retrieval_index[: num_train]
    train_data = data[train_index, :]
    train_targets = targets[train_index]
    query_data = data[query_index, :]
    query_targets = targets[query_index]
    retrieval_data = data[retrieval_index, :]
    retrieval_targets = targets[retrieval_index]

    return train_data, train_targets, \
           query_data, query_targets, \
           retrieval_data, retrieval_targets


def load_gist_data(root, num_train, num_query):
    """
    Load cifar-10 gist features.

    Args
        root(str): Path of dataset.
        num_train(int): Number of training data points.
        num_query(int): Number of query data points.

    Returns
        train_data(torch.Tensor): Training data points.
        train_targets(torch.Tensor): Training targets.
        query_data(torch.Tensor): Query data points.
        query_targets(torch.Tensor): Query targets.
        retrieval_data(torch.Tensor): Retrieval data points.
        retrieval_targets(torch.Tensor): Retrieval targets.
    """
    # Load matlab format data
    mat = sio.loadmat(root)

    # Concatenate
    train_data = mat['traindata']
    train_targets = mat['traingnd']
    query_data = mat['testdata']
    query_targets = mat['testgnd']
    data = np.concatenate((train_data, query_data), axis=0)
    targets = np.concatenate((train_targets, query_targets), axis=0).astype(np.int).squeeze()

    # Normalization
    data = (data - data.mean()) / data.std()

    # Split
    n = data.shape[0]
    perm_index = np.random.permutation(n)
    query_index = perm_index[:num_query]
    retrieval_index = perm_index[num_query:]
    train_index = retrieval_index[: num_train]
    train_data = data[train_index, :]
    train_targets = targets[train_index]
    query_data = data[query_index, :]
    query_targets = targets[query_index]
    retrieval_data = data[retrieval_index, :]
    retrieval_targets = targets[retrieval_index]

    return torch.from_numpy(train_data), \
           torch.from_numpy(train_targets), \
           torch.from_numpy(query_data), \
           torch.from_numpy(query_targets), \
           torch.from_numpy(retrieval_data), \
           torch.from_numpy(retrieval_targets)
