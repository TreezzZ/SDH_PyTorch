#!/usr/bin/env python
# -*- coding: utf-8 -*-

import data.dataloader as dataloader
import sdh

from data.transform import normalization
from data.transform import encode_onehot

import argparse
import torch
from loguru import logger


def run_sdh(opt):
    # Load data
    query_data, query_targets, \
    train_data, train_targets, \
    database_data, database_targets = dataloader.load_data(opt.data_path,
                                                           opt.dataset,
                                                           opt.num_query,
                                                           opt.num_train,
                                                           opt.batch_size,
                                                           opt.num_workers,
                                                           )

    # Normalization and one-hot
    query_data = normalization(query_data)
    query_data = torch.Tensor(query_data.reshape((query_data.shape[0], -1)))
    train_data = normalization(train_data)
    train_data = torch.Tensor(train_data.reshape((train_data.shape[0], -1)))
    database_data = normalization(database_data)
    database_data = torch.Tensor(database_data.reshape((database_data.shape[0], -1)))
    if opt.dataset == 'cifar10-gist':
        train_targets = torch.Tensor(encode_onehot(train_targets, 10))
        query_targets = torch.Tensor(encode_onehot(query_targets, 10))
        database_targets = torch.Tensor(encode_onehot(database_targets, 10))
    # elif opt.dataset == 'nus-wide':
    #     train_targets = torch.Tensor(train_dataloader.dataset.targets)
    #     query_targets = torch.Tensor(query_dataloader.dataset.targets)
    #     database_targets = torch.Tensor(database_dataloader.dataset.targets)

    # SDH Algorithm
    cl = [12, 24, 32, 64, 128]
    for c in cl:
        opt.code_length = c
        print(c)
        P, anchor = sdh.sdh(train_data,
                            train_targets,
                            query_data,
                            query_targets,
                            opt.code_length,
                            opt.num_anchor,
                            opt.max_iter,
                            opt.lamda,
                            opt.nu,
                            opt.sigma,
                            opt.topk,
                            opt.evaluate_freq,
                            )

        # Evaluate on whole dataset
        mAP = sdh.evaluate(query_data,
                           query_targets,
                           database_data,
                           database_targets,
                           anchor,
                           P,
                           opt.sigma,
                           topk=opt.topk,
                           )

        logger.info('final_mAP: {:.4f}'.format(mAP))


def load_parse():
    """加载程序参数

    Parameters
        None

    Returns
        opt: parser
        程序参数
    """
    parser = argparse.ArgumentParser(description='SDH_PyTorch')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='Dataset used to train (default: cifar10)')
    parser.add_argument('--data-path',
                        help='Path of cifar10 dataset')
    parser.add_argument('--code-length', default=12, type=int,
                        help='Binary hash code length (default: 12)')
    parser.add_argument('--max-iter', default=5, type=int,
                        help='Maximum iteration number (default: 5)')
    parser.add_argument('--num-anchor', default=1000, type=int,
                        help='Number of anchor points (default: 1000)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query(default: 1000)')
    parser.add_argument('--num-train', default=5000, type=int,
                        help='Number of train(default: 5000)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Compute map of top k (default: -1, use whole dataset)')
    parser.add_argument('--evaluate-freq', default=1, type=int,
                        help='Frequency of evaluate (default: 1)')

    parser.add_argument('--lamda', default=1, type=float,
                        help='Hyper-parameter, regularization term weight (default: 1.0)')
    parser.add_argument('--nu', default=1e-5, type=float,
                        help='Hyper-parameter, penalty term of hash function output (default: 1e-5)')
    parser.add_argument('--sigma', default=0.4, type=float,
                        help='Hyper-parameter, rbf kernel width (default: 0.4)')

    parser.add_argument('--gpu', default=0, type=int,
                        help='Use gpu (default: 0. -1: use cpu)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size (default: 128)')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of loading data workers (default: 0)')

    return parser.parse_args()


def set_seed(seed):
    import random
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


if __name__ == "__main__":
    opt = load_parse()
    logger.add('logs/file_{time}.log')

    # set_seed(20180707)

    if opt.gpu == -1:
        opt.device = torch.device("cpu")
    else:
        opt.device = torch.device("cuda:%d" % opt.gpu)

    logger.info(opt)

    run_sdh(opt)
