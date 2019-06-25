#!/usr/bin/env python
# -*- coding: utf-8 -*-


def calc_hamming_dist(B1, B2):
    """Compute hamming distance between B1 and B2

    Parameters
        B1, B2: Tensor
        hash code

    Returns
        dist: Tensor
        hamming distance
    """
    return 0.5 * (B2.shape[1] - B1 @ B2.t())
