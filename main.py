import gc

import numpy as np
import torch

from options.base_options import BaseOptions
from trainer import trainer
from utils import set_seed, print_args, overwrite_with_yaml
from collections import defaultdict


def main(args):
    overall_stats = defaultdict(list)
    # list_test_acc = []
    # list_valid_acc = []
    # list_train_loss = []
    if args.compare_model:
        args = overwrite_with_yaml(args, args.type_model, args.dataset)
    print_args(args)
    for seed in range(args.N_exp):
        print(f'seed (which_run) = <{seed}>')
        args.random_seed = seed
        set_seed(args)
        torch.cuda.empty_cache()
        trnr = trainer(args, seed)
        stats = trnr.train_and_test()
        for k, v in stats.items():
            overall_stats[k].append(v)
        #     if args.task == 'node':
        #     list_test_acc.append(stats['test_acc'])
        #     list_valid_acc.append(stats['valid_acc'])
        #     list_train_loss.append(stats['loss_train'])
        # else:
        #     list_test_acc.append(stats['test_acc'])
        #     list_valid_acc.append(stats['valid_acc'])
        #     list_train_loss.append(stats['loss_train'])

        del trnr
        torch.cuda.empty_cache()
        gc.collect()

        # record training data
        msg = 'mean and std of all stats: '
        for k, v in overall_stats.items():
            msg = msg + f"{k}: {np.mean(v):.4f}:{np.std(v):.4f}, "
        print(msg)
        # print('mean and std of test acc: {:.4f}±{:.4f}'.format(
        #     np.mean(list_test_acc), np.std(list_test_acc)))

    msg = f'final mean and std of test acc with <{args.N_exp}> runs:'
    for k, v in overall_stats.items():
        msg = msg + f"{k}: {np.mean(v):.4f}:{np.std(v):.4f}, "
    print(msg)

if __name__ == "__main__":
    args = BaseOptions().initialize()
    main(args)
