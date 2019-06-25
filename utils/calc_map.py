#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.calc_hamming_dist import calc_hamming_dist

import torch


def calc_map(query_code,
             database_code,
             query_labels,
             database_labels,
             device,
             topk=None,
             ):
    """Compute mAP

    Parameters
        query_code: ndarray, {-1, +1}^{m * Q}
        Query hash code

        database_code: ndarray, {-1, +1}^{n * Q}
        Database hash code

        query_labels: ndarray, {0, 1}^{m * n_classes}
        Query label，onehot

        database_labels: ndarray, {0, 1}^{n * n_classes}
        Database label，onehot

        topk: int
        Compute mAP using top k retrieval result

    Returns
        meanAP: float
        Mean Average Precision
    """
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieval result
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Hamming distance
        hamming_dist = calc_hamming_dist(query_code[i, :], database_code)

        # According to hamming distance, sort and acquire topk data
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Number of retrieval result
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieval image
        if retrieval_cnt == 0:
            continue

        # Generate score by position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # index of retrieval result
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP


