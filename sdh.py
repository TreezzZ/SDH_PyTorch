#!/usr/bin/env python
# -*- coding: utf-8 -*-

import utils.dcc as dcc

from utils.calc_map import calc_map
from data.transform import encode_onehot

import torch

from sklearn.metrics.pairwise import rbf_kernel


def sdh(train_data,
        train_targets,
        query_data,
        query_targets,
        code_length,
        num_class,
        num_anchor,
        max_iter,
        lamda,
        nu,
        sigma,
        evaluate_freq,
        ):
    """SDH algorithm
    
    Parameters
        train_data: Tensor
        Training data

        train_targets: Tensor
        Training targets

        query_data: Tensor
        Query data

        query_targets: Tensor
        Query targets

        code_length: int
        Hash code length

        num_class: int
        Number of classes

        num_anchor: int
        Number of anchor points
        
        max_iter: int
        Maximum iteration number

        lamda: float
        Hyper-parameter

        nu: float
        Hyper-parameter

        sigma: float
        Hyper-parameter

        evaluate_freq: int
        Frequency of evaluate

    Returns
        hash_code: Tensor
        Binary code
    """
    # Permute data
    perm_index = torch.randperm(train_data.shape[0])
    train_data = train_data[perm_index, :]
    train_targets = train_targets[perm_index, :]

    # Randomly select num_anchor samples from the trainning data
    anchor = train_data[torch.randperm(train_data.shape[0])[:num_anchor], :]

    # Map training data via RBF kernel
    phi_x = torch.Tensor(rbf_kernel(train_data.numpy(), anchor.numpy(), sigma))

    # Initialize B
    B = torch.randn((code_length, train_data.shape[0])).sign()
    Y = train_targets

    for itr in range(max_iter):
        # G-Step
        W = torch.inverse(B @ B.t() + lamda * torch.eye(code_length)) @ B @ Y

        # F-Step
        P = (torch.inverse(phi_x @ phi_x.t()) @ phi_x).t() @ B.t()
        F_X = phi_x @ P

        # B-Step
        B = dcc.solve(B, W, Y, F_X, nu)

        if itr % evaluate_freq == evaluate_freq - 1:
            mAP = evaluate(query_data, query_targets, train_data, train_targets, anchor, P, sigma, 5000)
            print('map: {:.4f}'.format(mAP))
    
    return B, P, anchor
   

def evaluate(query_data, query_targets, database_data, database_targets, anchor, P, sigma, topk=None):
    """Evaluate Algorithm

    Parameters
        query_data: Tensor
        Query data

        query_targets: Tensor
        Query targets

        database_data: Tensor
        Database data

        database_targets: Tensor
        Database targets

        anchor: Tensor
        Anchor points

        P: Tensor
        Projection matrix

        sigma: float
        RBF kernel width

        topk: int
        Compute mAP using top k retrieval result

    Returns
        meanAP: float
        mean Average precision
    """
    # Generate database hash code and query hash code
    database_code = generate_code(database_data, anchor, P, sigma)
    query_code = generate_code(query_data, anchor, P, sigma)

    # 计算map
    meanAP = calc_map(query_code,
                      database_code,
                      query_targets,
                      database_targets,
                      'cpu',
                      topk,
                      )

    return meanAP


def generate_code(data, anchor, P, sigma):
    """generate hash code from data using projection matrix

    Parameters
        data: Tensor
        Data

        anchor: Tensor
        Anchor points

        P: Tensor
        Projection matrix

        sigma: float
        RBF kernel width

    Returns
        code: Tensor
        hash code
    """
    return (torch.Tensor(rbf_kernel(data.numpy(), anchor.numpy(), sigma)) @ P).sign()




