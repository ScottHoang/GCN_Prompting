import collections
import importlib
import os
import random
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator

import utils
from Dataloader import load_data, load_ogbn, prepare_data
from tricks import TricksComb  # TricksCombSGC
from utils import AcontainsB
from utils import TaskPredictor
from utils import shortest_path
from utils import CompoundOptimizers
from utils import StratifiedSampler
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler
from copy import deepcopy
from prompt.prompt import *
from tasks.edge_task import create_edge_task
from tasks.node_task import create_node_task
from tasks.transfer_task import create_domain_transfer_task
from tasks.baseline_node_task import create_node_task as create_nodeBaseLine_task
from tasks.vgae_task import create_vgae_task, create_vge_node_transfer_task
from tasks.pretraining_task import create_pretrain_task
from models import VGAE
import matplotlib.pyplot as plt
import time
import datetime
import logging
BASIC = ['node', 'edge']
DOMAIN = ['dt']
VGAE = ['vgae']


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


class trainer(object):
    def __init__(self, args, which_run):
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.logger = logging.getLogger()
        self.args = args
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        self.which_run = which_run
        if args.task != 'node':
            self.num_classes = args.num_classes
            args.num_classes = args.dim_hidden
            self.edge_predictor = TaskPredictor(args.dim_hidden, args.dim_hidden, 1, 2,
                                                args.dropout, lr=0.0005, weight_decay=self.weight_decay).to(self.device)
            self.optimizer_edge = self.edge_predictor.optimizer
        if args.type_model != 'mlp':
            if args.compare_model or args.type_model == 'VGAE' :  # only compare model
                Model = getattr(importlib.import_module("models"), self.type_model)
                self.model = Model(args)
            else:  # compare tricks combinations
                self.model = TricksComb(args)
        else:
            self.model = TaskPredictor(args.num_feats, args.dim_hidden, self.dim_hidden, args.num_layers,
                                           args.dropout, lr=self.args.lr, weight_decay=self.weight_decay).to(
                self.device)
        self.model.to(self.device)
        self.optimizer = self.model.optimizer
        self.run_iter = 0

    def get_channel_by_task(self, task):
        args = self.args
        dim_hidden = args.dim_hidden
        if self.type_model == 'VGAE':
            dim_hidden = dim_hidden // 2
        in_c = dim_hidden
        if self.args.prompt_k:
            if self.prompt_raw:
                raw_c = self.data.x.size(1)
                if args.prompt_aggr in ['concat']:
                    raw_c *= args.prompt_k
                in_c = dim_hidden + raw_c
            else:
                if args.prompt_aggr in ['concat']:
                    in_c *= args.prompt_k
                    in_c += dim_hidden
                elif self.prompt_aggr in ['att', 'edges']:
                    pass
                else:
                    in_c += dim_hidden
        if args.prompt_w_org_features:
            in_c += self.data.x.size(-1)
        return in_c
        # return self.data.x.size(-1)

    def init_pretrain(self):
        args = self.args
        edge_decoder = attr_decoder = None
        lr = args.lr
        args.lr = args.prompt_pretrain_lr
        if args.type_model != 'mlp':
            if args.compare_model:  # only compare model
                Model = getattr(importlib.import_module("models"), self.type_model)
                model = Model(args)
            else:  # compare tricks combinations
                model = TricksComb(args)
        else:
            model = TaskPredictor(args.num_feats, args.dim_hidden, self.dim_hidden, args.num_layers,
                                       args.dropout, lr=self.args.lr, weight_decay=self.weight_decay)
        model.to(self.device)
        args.lr = lr
        if utils.AcontainsB(self.prompt_pretrain_type, ['edgeMask']):
            edge_decoder = TaskPredictor(args.dim_hidden, args.dim_hidden, 1, 2,
                                                args.dropout, lr=self.prompt_pretrain_lr, weight_decay=self.weight_decay).to(self.device)
        if utils.AcontainsB(self.prompt_pretrain_type, ['attrMask']):
            attr_decoder = TaskPredictor(args.dim_hidden, args.dim_hidden, self.prompt_pca, 2,
                                         args.dropout, lr=self.prompt_pretrain_lr, weight_decay=self.weight_decay).to(self.device)
        return model, edge_decoder, attr_decoder

    def init_predictor_by_type(self, type, in_c, args=None):
        num_classes = self.num_classes
        if self.downstream_task == 'edge':
            num_classes = args.prompt_dim_hidden

        if type == 'mlp':
            predictor = TaskPredictor(in_c, args.prompt_dim_hidden, num_classes, args.prompt_layer,
                                           args.dropout, lr=self.args.lr, weight_decay=self.weight_decay).to(self.device)
            optimizer = predictor.optimizer
        else:
            Model = getattr(importlib.import_module("models"), self.prompt_head)
            #
            args.num_feats = in_c
            args.num_layers = self.args.prompt_layer
            args.num_classes = num_classes
            args.dim_hidden = args.prompt_dim_hidden
            if not self.prompt_trick:
                predictor = Model(args).to(self.device)
            else:
                predictor = TricksComb(args, self.prompt_head).to(self.device)
            optimizer = predictor.optimizer

        ####
        if self.downstream_task == 'edge':
            self.model = predictor
            predictor = TaskPredictor(args.prompt_dim_hidden, args.prompt_dim_hidden, 1, 2,
                                         args.dropout, lr=self.lr, weight_decay=self.weight_decay).to(self.device)
            optimizer = predictor.optimizer


        return predictor, optimizer

    def init_predictor_by_tasks(self, task):
        args = deepcopy(self.args)
        in_c = self.get_channel_by_task(task)
        return self.init_predictor_by_type(args.prompt_head, in_c, args)

    def set_dataloader(self):
        self.split_idx = None
        if self.dataset == 'ogbn-arxiv':
            self.data, self.split_idx = load_ogbn(self.dataset)
            self.train_idx = self.split_idx['train']
            self.evaluator = Evaluator(name='ogbn-arxiv')
        else:
            self.data = load_data(self.dataset, self.which_run, norm=None)
            # self.data = prepare_edge_data(self.args, self.which_run).to(self.device)

            if self.args.task in ['dtbfs']:
                self.sampler = GraphSAINTRandomWalkSampler(self.data, batch_size=1, num_steps=self.data.x.size(0),
                                                           walk_length=self.args.prompt_k)

        # elif self.args.task in ['edge', 'vgae', 'dt', 'dtbfs', 'dtzero', 'prompt', 'nodebs', 'dtvgae']:
        self.data = prepare_data(self.data, self.args, self.which_run).to(self.device)

    def train_and_test(self):
        if self.args.task in ['node', 'edge']:
            self.set_dataloader()
            if self.args.task == 'node':
                learner = create_node_task(self.args, self.model, self.args.batch_size, self.data, self.dataset, self.type_trick,
                                           self.split_idx)
            else:
                self.args.task = 'edge'
                self.set_dataloader()
                if self.type_model == 'VGAE':
                    learner = create_vgae_task(self.model, self.args.batch_size, self.data, self.prompt_temp)
                else:
                    model, edge_decoder, attr_decoder = self.init_pretrain()
                    self.model = model
                    learner = create_pretrain_task(self.model, self.batch_size, self.data, self.args, 1.0,
                                                            edge_predictor=edge_decoder,
                                                            attr_predictor=attr_decoder)
                    # edge_learner = create_edge_task(self.model, self.edge_predictor, self.args.batch_size, self.data, self.prompt_temp)

            train_fn = lambda: self.sequential_run(learner.task_train, learner.task_test)
            stats, all_stats = self.train_test_frame(train_fn, stats_fn=learner.stats, epochs=self.epochs)

            return stats

        elif self.args.task in ['dt', 'dtvgae']:
            task = self.args.task
            ########################### pretrain
            self.args.task = 'edge'
            self.set_dataloader()
            if self.type_model == 'VGAE':
                pretrain_learner = create_vgae_task(self.model, self.args.batch_size, self.data, self.prompt_temp)
            else:
                model, edge_decoder, attr_decoder = self.init_pretrain()
                self.model = model
                pretrain_learner = create_pretrain_task(self.model, self.batch_size, self.data, self.args, 1.0, edge_predictor=edge_decoder,
                                                        attr_predictor=attr_decoder)
                # edge_learner = create_edge_task(self.model, self.edge_predictor, self.args.batch_size, self.data, self.prompt_temp)
            train_fn = lambda: self.sequential_run(pretrain_learner.task_train, pretrain_learner.task_test)
            pretrain_stats, _ = self.train_test_frame(train_fn, pretrain_learner.stats, self.epochs)
            ########################### predictor training
            # init predictor
            self.args.task = task
            #
            if task in ['dt', 'dtvgae']:
                if self.type_model == 'VGAE':
                    self.model.eval()
                    _, mu, logvar = self.model(self.data.x, self.data.edge_index)
                    self.prompt_embs = Embeddings(mu.detach().clone(), self.data.x.clone(),
                                                  lr=self.prompt_lr, weight_decay=self.weight_decay, epochs=self.epochs)
                    self.prompt_embs.mu = mu
                    self.prompt_embs.logvar = logvar
                else:
                    embs = self.model(self.data.x, self.data.edge_index)
                    self.prompt_embs = Embeddings(embs.detach().clone(), self.data.x.clone(),
                                                  lr=self.prompt_lr, weight_decay=self.weight_decay, epochs=self.epochs,
                                                  )
                self.predictor, optimizer  = self.init_predictor_by_tasks(task)
            else:
                raise ValueError

            self.prompt_embs.to(self.device)
            init_fn = create_domain_transfer_task if task == 'dt' else create_vge_node_transfer_task
            learner = init_fn(self.model, self.predictor, self.prompt_embs, self.args, task,
                              self.data,
                              self.split_idx)

            ######## the SP matrix:
            distance = torch.load(f'data/{self.dataset}/distance.pth').float()
            distance[distance.eq(510)] = -1
            distance[distance.eq(-1)] = distance.max() + 1
            distance.add_(
                1.0)  # avoid log(1) = 0, instead log(2) = 0.69 is good for punishing node outside of desired range, while not excluding them completely.
            distance.fill_diagonal_(1)  # log(diag) = 0
            distance.add_(distance.t())
            distance.div_(2.0)
            #####
            learner.model.log_distance = torch.log(distance).div_(self.prompt_distance_temp).to(self.device)
            # train /test fn init
            train_fn = lambda: self.sequential_run(learner.task_train, learner.task_test)
            stats, all_stats = self.train_test_frame(train_fn, stats_fn=learner.stats, epochs=self.epochs)
            # stats.update(pretrain_stats)

            internal_stats = learner.model.stats
            if self.plot_info:
                self.plot_figures(internal_stats, all_stats)
            return stats
        else:
            raise ValueError

        if self.prompt_save_embs:
            pass

    def train_test_frame(self, runs_fn, stats_fn, epochs=1000):
        best_stats = None
        all_stats = collections.defaultdict(list)
        patience = self.args.patience
        bad_counter = 0.
        # val_loss_history = []
        for epoch in range(epochs):
            stats = runs_fn()
            for k, v in stats.items():
                all_stats[k].append(v)
            best_stats = stats if best_stats is None else best_stats
            best_stats, bad_counter = stats_fn(best_stats, stats, bad_counter)
            if bad_counter == patience:
                break
            if epoch % 20 == 0 or epoch == 1:
                log = f"Epoch: {epoch:03d}, "
                for k, v in best_stats.items():
                    log = log + f"{k}: {v:.4f}, "
                self.logger.info(log)
        log = ''
        self.logger.info(log)
        return best_stats, all_stats

    @staticmethod
    def sequential_run(*args):
        stats = {}
        for r in args:
            stats.update(r())
        return stats

    def get_bfs_prompts(self):
        target_k = self.args.prompt_k+1
        all_prompts = []
        for i in range(self.data.num_nodes):
            prompt = self.bfs(i, target_k)
            prompt.sort()
            # prompts.append(i)
            prompt = torch.tensor(prompt)
            if prompt.size(0) < target_k:
                prompt = prompt.tile(target_k // prompt.size(0) + 2)[:target_k]
            all_prompts.append(prompt.unsqueeze(0))
        all_prompts = torch.cat(all_prompts, dim=0)
        return all_prompts

    def bfs(self, node, target):
        prompt=set([])
        queue=[node]
        seen = set()

        while queue:
            s = queue.pop(0)
            if s not in seen:
                neighbors = self.data.edge_index[1, self.data.edge_index[0, :].eq(s)].tolist()
                random.shuffle(neighbors)
                for n in neighbors:
                    prompt.add(n)
                    queue.append(n)
                    if len(prompt) == target*3:
                        break
            if len(prompt) == target*3:
                break
            seen.add(s)
        if len(prompt) > target:
            prompt = random.sample(prompt, target)
        return list(prompt)

    def get_mad(self, embs):
        n = self.data.x.size(0)
        tgt = torch.ones(n, n).to(embs.device)
        return utils.MAD(embs, tgt)

    def plot_figures(self, distances, stats):
        root = os.path.join('figures', self.prompt_mode, self.dataset,
                            datetime.datetime.now().strftime('%y-%m-%d||%H:%M')
                            )
        if not os.path.isdir(root):
            os.makedirs(root)

        fig, axes = plt.subplots(len(stats)+2, 1, figsize=(10, 10))
        for i, (y_label, values) in enumerate(stats.items()):
            ax = axes[i]
            values = np.array(values)
            ax.plot(np.arange(values.shape[0]), values)
            ax.set_ylabel(y_label)
            # ax.set_xlabel(x_label)
        axes[-2].plot(np.arange(len(distances['mean_distance'])), distances['mean_distance'])
        axes[-2].set_ylabel('mean_distance')
        axes[-1].plot(np.arange(len(distances['mean_i2nr'])), distances['mean_i2nr'])
        axes[-1].set_ylabel('mean_i2nr')

        plt.savefig(os.path.join(root, 'img.png'))
        torch.save({'distance': distances, 'stats': stats}, os.path.join(root, 'data.pth'))









