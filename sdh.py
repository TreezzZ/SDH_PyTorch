#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.calc_map import calc_map

import torch

from sklearn.metrics.pairwise import rbf_kernel
from loguru import logger


def sdh(train_data,
        train_targets,
        query_data,
        query_targets,
        code_length,
        num_anchor,
        max_iter,
        lamda,
        nu,
        sigma,
        topk,
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

        topk: int
        Compute mAP using top k retrieval result

        evaluate_freq: int
        Frequency of evaluate

    Returns
        final_P, final_anchor: Tensor
        Used to construct F(X)
    """
    # Permute data
    perm_index = torch.randperm(train_data.shape[0])
    train_data = train_data[perm_index, :]
    train_targets = train_targets[perm_index, :]

    # Randomly select num_anchor samples from the trainning data
    anchor = train_data[torch.randperm(train_data.shape[0])[:num_anchor], :]

    # Map training data via RBF kernel
    phi_x = torch.Tensor(rbf_kernel(train_data.numpy(), anchor.numpy(), sigma)).t()

    # Initialize B
    B = torch.randn((code_length, train_data.shape[0])).sign()
    Y = train_targets.t()

    best_map = 0.0
    for itr in range(max_iter):
        # G-Step
        W = torch.inverse(B @ B.t() + lamda * torch.eye(code_length)) @ B @ Y.t()

        # F-Step
        P = torch.inverse(phi_x @ phi_x.t()) @ phi_x @ B.t()
        F_X = P.t() @ phi_x

        # B-Step
        B = solve_dcc(B, W, Y, F_X, nu)

        # Evaluate within iteration
        if itr % evaluate_freq == evaluate_freq - 1:
            mAP = evaluate(query_data, query_targets, train_data, train_targets, anchor, P, sigma, topk)
            logger.info('[iter: {}][mAP: {:.4f}]'.format(itr+1, mAP))
            if best_map < mAP:
                best_map = mAP
                final_P = P
                final_anchor = anchor

    return final_P, final_anchor


def solve_dcc(B, W, Y, F_X, nu):
    """Solve DCC(Discrete Cyclic Coordinate Descent) problem
    """
    for i in range(B.shape[0]):
        Q = W @ Y + nu * F_X

        q = Q[i, :]
        v = W[i, :]
        W_prime = torch.cat((W[:i, :], W[i+1:, :]))
        B_prime = torch.cat((B[:i, :], B[i+1:, :]))

        B[i, :] = (q - B_prime.t() @ W_prime @ v).sign()

    return B
   

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

    # Compute mAP
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




