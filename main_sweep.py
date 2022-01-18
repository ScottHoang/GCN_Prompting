import copy
import gc
import os

import numpy as np
import torch

from options.base_options import BaseOptions
from trainer import trainer
from utils import set_seed, print_args, overwrite_with_yaml
from collections import defaultdict
import wandb
from sweep_config.sweep_params import parameters_dict

global args
def main(args):
    overall_stats = defaultdict(list)
    if not os.path.isdir('embeddings'):
        os.mkdir('embeddings')
    if args.compare_model:
        args = overwrite_with_yaml(args, args.type_model, args.dataset)
    # print_args(args)
    for seed in range(args.N_exp):
        print(f'seed (which_run) = <{seed}>')
        args.random_seed = seed
        set_seed(args)
        torch.cuda.empty_cache()
        trnr = trainer(args, seed)
        stats = trnr.train_and_test()
        for k, v in stats.items():
            overall_stats[k].append(v)

        del trnr
        torch.cuda.empty_cache()
        gc.collect()

        # record training data
        msg = 'mean and std of all stats: '
        for k, v in overall_stats.items():
            msg = msg + f"{k}: {np.mean(v):.4f}:{np.std(v):.4f}, "
        print(msg)

    mean_stats = {}
    for k, v in overall_stats.items():
        mean_stats[k] = np.mean(v)
    return mean_stats


def sweep():
    sweep_config = {'method': 'bayes'}
    metric = {'name': 'test_acc', 'goal': 'maximize'}
    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config,
                           project=f'sweep-{args.dataset}-5')
    wandb.agent(sweep_id, function=run_sweep)

def run_sweep():
    global args
    with wandb.init(config=None):
        local_args = copy.deepcopy(args)
        for k, v in wandb.config.items():
            setattr(local_args, k, v)
        stats = main(local_args)
        wandb.log(stats)
        gc.collect()


if __name__ == "__main__":
    global args
    args = BaseOptions().initialize()
    sweep()