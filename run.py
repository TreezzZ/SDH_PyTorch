import argparse
import torch
import random
import numpy as np
import sdh

from loguru import logger
from data.data_loader import load_data


def run():
    # Load configuration
    args = load_config()
    logger.add('logs/{}_code_{}_anchor_{}_train_{}_lamda_{}_nu_{}_sigma_{}_topk_{}.log'.format(
            args.dataset,
            '_'.join([str(code_length) for code_length in args.code_length]),
            args.num_anchor,
            args.num_train,
            args.lamda,
            args.nu,
            args.sigma,
            args.topk,
        ),
        rotation='500 MB',
        level='INFO',
    )
    logger.info(args)

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    train_data, train_targets, query_data, query_targets, retrieval_data, retrieval_targets = load_data(
        args.dataset,
        args.root,
        args.num_train,
        args.num_query,
    )

    # Training
    for code_length in args.code_length:
        checkpoint = sdh.train(
            train_data,
            train_targets,
            query_data,
            query_targets,
            retrieval_data,
            retrieval_targets,
            code_length,
            args.num_anchor,
            args.max_iter,
            args.lamda,
            args.nu,
            args.sigma,
            args.device,
            args.topk,
            args.evaluate_interval,
        )
        logger.info('[code length:{}][map:{:.4f}]'.format(code_length, checkpoint['map']))

        # Save checkpoint
        torch.save(checkpoint, 'checkpoints/{}_code_{}_anchor_{}_train_{}_lamda_{}_nu_{}_sigma_{}_topk_{}_map_{:.4f}.pt'.format(
            args.dataset,
            code_length,
            args.num_anchor,
            args.num_train,
            args.lamda,
            args.nu,
            args.sigma,
            args.topk,
            checkpoint['map'],
        ))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='SDH_PyTorch')
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--root',
                        help='Path of dataset')
    parser.add_argument('--code-length', default='12,16,24,32,48,64,128', type=str,
                        help='Binary hash code length.(default: 12,16,24,32,48,64,128)')
    parser.add_argument('--max-iter', default=5, type=int,
                        help='Number of iterations.(default: 5)')
    parser.add_argument('--num-anchor', default=1000, type=int,
                        help='Number of anchor.(default: 1000)')
    parser.add_argument('--num-train', default=5000, type=int,
                        help='Number of training data points.(default: 5000)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--seed', default=3367, type=int,
                        help='Random seed.(default: 3367)')
    parser.add_argument('--evaluate-interval', default=1, type=int,
                        help='Evaluation interval.(default: 1)')
    parser.add_argument('--lamda', default=1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--nu', default=1e-5, type=float,
                        help='Hyper-parameter.(default: 1e-5)')
    parser.add_argument('--sigma', default=3e-4, type=float,
                        help='Hyper-parameter. 2e-3 for cifar-10-gist, 3e-4 for others.')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))

    return args


if __name__ == '__main__':
    run()
