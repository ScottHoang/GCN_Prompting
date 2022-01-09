import gc
import os

import numpy as np
import torch

from options.base_options import BaseOptions
from trainer import trainer
from utils import set_seed, print_args, overwrite_with_yaml
from collections import defaultdict
import wandb

args = None

def main(args):
    overall_stats = defaultdict(list)
    if not os.path.isdir('embeddings'):
        os.mkdir('embeddings')
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
        if args.prompt_save_embs:
            # pkgs.update({
            # 'labels': trnr.data.y,
            # 'edge_index': trnr.data.edge_index})
            save_prompt_embs(trnr, args, seed, stats)
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

    msg = f'final mean and std of test acc with <{args.N_exp}> runs:'
    for k, v in overall_stats.items():
        msg = msg + f"{k}: {np.mean(v):.4f}:{np.std(v):.4f}, "
    print(msg)
    return overall_stats

def save_prompt_embs(trnr, args, seed, stats):
    assert args.task not in ['node', 'edge']
    name = f'{trnr.dataset}_{args.task}_{args.prompt_head}_{args.num_layers}_{args.prompt_layer}'
    if args.prompt_w_org_features:
        name = name + "_org"
    name = name + '.pth'
    path = os.path.join('embeddings', name)
    pkg = {"static_embs": trnr.prompt_embs.static_embs,
           "learned_embs": trnr.prompt_embs.embs,
           'train_edges': trnr.data.train_pos,
           'val_edges': trnr.data.val_pos,
           'test_edges': trnr.data.test_pos,
           'prompt': trnr.bfs_prompts}
    pkg.update(stats)
    if os.path.isdir(path):
        pkgs = torch.load(path)
    else:
        pkgs = {
            'labels': trnr.data.y,
            'edge_index': trnr.data.edge_index}
    pkgs[seed] = pkg
    torch.save(pkgs, path)



def sweep():

    args.run_iters = 1
    # exec(f'from config.sweep_files.{args.sweep_config} import parameters_dict as sweep_params')
    sweep_config = {'method': 'grid'}
    metric = {'name': 'test_loss_tune', 'goal': 'minimize'}
    # metric = {'name': 'test_acc_tune' , 'goal': 'maximize'}
    sweep_config['metric'] = metric
    sweep_config['parameters'] = sweep_params
    sweep_id = wandb.sweep(sweep_config,
                           project=f'sweep-nas-gnn-{args.sweep_id}')
    wandb.agent(sweep_id, function=run_sweep)

def run_sweep():
    with wandb.init(config=None):
        main(args)
        gc.collect()


if __name__ == "__main__":
    global args
    args = BaseOptions().initialize()
    sweep()
