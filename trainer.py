import collections
import importlib
import random

import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator

from Dataloader import load_data, load_ogbn, prepare_edge_data
from tricks import TricksComb  # TricksCombSGC
from utils import AcontainsB
from utils import TaskPredictor
from utils import CompoundOptimizers
from utils import StratifiedSampler
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler
from copy import deepcopy
from prompt.prompt import *
from tasks.edge_task import create_edge_task
from tasks.node_task import create_node_task
from tasks.transfer_task import create_domain_transfer_task


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


class trainer(object):
    def __init__(self, args, which_run):
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.args = args
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        self.which_run = which_run
        # set dataloader
        self.set_dataloader()
        #
        if args.task != 'node':
            self.num_classes = args.num_classes
            args.num_classes = args.dim_hidden
            self.edge_predictor = TaskPredictor(args.dim_hidden, args.dim_hidden, 1, 3,
                                                args.dropout, lr=0.0005, weight_decay=self.weight_decay).to(self.device)
            self.optimizer_edge = self.edge_predictor.optimizer
            # self.node_predictor, self.optimizer_node = self.init_node_predictor()

        if args.compare_model:  # only compare model
            Model = getattr(importlib.import_module("models"), self.type_model)
            self.model = Model(args)
        else:  # compare tricks combinations
            self.model = TricksComb(args)
        self.model.to(self.device)
        self.optimizer = self.model.optimizer

    def init_node_predictor(self):
        args = deepcopy(self.args)
        if self.args.task != 'dt':
            in_c = args.dim_hidden
            if args.prompt_aggr == 'concat':
                in_c *= (args.prompt_k + 1)
            if args.prompt_w_org_features:
                in_c += self.data.x.size(-1)
            if self.args.task == 'dtbfs':
                in_c += self.args.dim_hidden
        else:
            in_c = self.args.dim_hidden

        if args.prompt_head == 'mlp':
            node_predictor = TaskPredictor(in_c, args.dim_hidden, self.num_classes, args.prompt_layer,
                                           args.dropout, lr=self.args.lr, weight_decay=self.weight_decay).to(self.device)
            optimizer_node = node_predictor.optimizer
        elif args.prompt_head == 'gnn':
            Model = getattr(importlib.import_module("models"), self.type_model)
            #
            args.num_feats = in_c
            args.num_layers = self.args.prompt_layer
            args.num_classes = self.num_classes
            #
            node_predictor = Model(args).to(self.device)
            optimizer_node = node_predictor.optimizer
        else:
            raise NotImplementedError

        return node_predictor, optimizer_node

    def init_auto_prompt(self):

        # candidates = self.get_prompt_candidates()

        candidates = torch.randint(0, self.data.x.size(0), (self.args.prompt_k,))
        node_embeddings = Embeddings(self.model(self.data.x, self.data.edge_index))
        node_embeddings_grad = GradientStorage(node_embeddings)

        return candidates, node_embeddings, node_embeddings_grad
        # grad = embedding_grad.get()

    def set_dataloader(self):
        if self.args.task in ['node', 'dt', 'dtbfs', 'dtzero', 'prompt']:
            if self.dataset == 'ogbn-arxiv':
                self.data, self.split_idx = load_ogbn(self.dataset)
                self.train_idx = self.split_idx['train'].to(self.device)
                self.evaluator = Evaluator(name='ogbn-arxiv')
                self.loss_fn = torch.nn.functional.nll_loss
            else:
                self.data = load_data(self.dataset, self.which_run)
                self.split_idx=None
                # self.data = prepare_edge_data(self.args, self.which_run).to(self.device)
                self.loss_fn = torch.nn.functional.nll_loss

            if self.args.task in ['dtbfs']:
                self.sampler = GraphSAINTRandomWalkSampler(self.data, batch_size=1, num_steps=self.data.x.size(0),
                                                           walk_length=self.args.prompt_k)
            self.data.to(self.device)
        elif self.args.task == 'edge':
            self.data = prepare_edge_data(self.args, self.which_run).to(self.device)
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            pass

    def train_and_test(self):
        if self.args.task in ['node', 'edge']:
            if self.args.task == 'node':
                learner = create_node_task(self.model, self.args.batch_size, self.data, self.dataset, self.type_trick,
                                           self.split_idx)
            else:
                learner = create_edge_task(self.model, self.edge_predictor, self.args.batch_size, self.data)

            train_fn = lambda: self.sequential_run(learner.task_train, learner.task_test)
            stats = self.train_test_frame(train_fn, stats_fn=learner.stats)

            if self.args.task == 'edge':
                stats.update(self.edge2node_prediction())
            return stats
        elif self.args.task in ['dt', 'dtbfs','dtzero']:
            task = self.args.task
            ########################### pretrain
            self.args.task = 'edge'
            self.set_dataloader()
            edge_learner = create_edge_task(self.model, self.edge_predictor, self.args.batch_size, self.data)
            train_fn = lambda: self.sequential_run(edge_learner.task_train, edge_learner.task_test)
            pretrain_stats = self.train_test_frame(train_fn, edge_learner.stats)
            ########################### predictor training
            # init predictor
            self.args.task = task
            self.node_predictor, optimizer_node = self.init_node_predictor()
            #
            if task in ['dt', 'dtbfs']:
                self.prompt_embs = Embeddings(self.model(self.data.x, self.data.edge_index).detach().clone(),
                                              lr=self.prompt_lr, weight_decay=self.weight_decay)
            elif task == 'dtzero':
                self.prompt_embs = Embeddings(self.model(self.data.x, self.data.edge_index).detach().clone(), init_as_zero=True,
                                              prompt_size=self.args.prompt_k, lr=self.prompt_lr, weight_decay=self.weight_decay)

            self.prompt_embs.to(self.device)
            #
            if task in 'dtbfs':
                self.bfs_prompts = self.get_bfs_prompts()
            else:
                self.bfs_prompts = None
            #
            learner = create_domain_transfer_task(self.model, self.node_predictor, self.prompt_embs, task, self.args.prompt_aggr,
                                                  self.args.prompt_w_org_features, self.bfs_prompts, self.data, self.dataset,
                                                  self.args.batch_size, self.type_trick, self.split_idx)
            # train /test fn init
            train_fn = lambda: self.sequential_run(learner.task_train, learner.task_test)
            stats = self.train_test_frame(train_fn, stats_fn=learner.stats)
            stats.update(pretrain_stats)
            return stats
        elif self.args.task == 'prompt':
            # pretrain
            task = self.args.task
            self.args.task = 'edge'
            self.set_dataloader()
            stat_fn = self.edge_stats
            train_fn = lambda: self.sequential_run(self.run_trainSet, self.run_testSet)
            pretrain_stats = self.train_test_frame(train_fn, stat_fn)
            #########
            self.args.task = task
            self.loss_fn = torch.nn.functional.nll_loss
            # self.set_dataloader()
            self.prompt_candidates, self.node_embeddings, self.node_embeddings_grad = self.init_auto_prompt()
            train_fn = lambda: self.sequential_run(self.prompt_search, self.prompt_test)
            stat_fn = self.prompt_stats
            prompt_stats = self.train_test_frame(train_fn, stat_fn)
            return prompt_stats

    def prompt_search(self):
        prompt = self.prompt_candidates.tile(self.data.num_nodes).view(self.data.num_nodes, -1)
        emb = self.node_embeddings.embs
        accuracy, logits = self.edge2node_logits('train', prompt, emb)

        loss = self.loss_fn(F.log_softmax(logits, dim=-1), self.data.y[self.data.train_mask])
        loss.backward()
        grad = emb.grad
        token_to_flip_idx = random.randrange(self.prompt_candidates.size(0))
        token_to_flip = self.prompt_candidates[token_to_flip_idx]
        candidates = hotflip_attack(grad[token_to_flip].unsqueeze(0), emb, num_candidates=100)
        avg_grad = 0
        for c in candidates:
            new_prompt = self.prompt_candidates.clone()
            new_prompt[token_to_flip_idx] = c
            new_prompt = new_prompt.tile(self.data.num_nodes).view(self.data.num_nodes, -1)
            accuracy, logits = self.edge2node_logits('train', new_prompt, emb)
            avg_grad += emb.grad

        avg_grad /= 100
        candidates = hotflip_attack(avg_grad[token_to_flip].unsqueeze(0), emb, num_candidates=20)
        accuracy, logits = self.edge2node_logits('val', prompt, emb)
        cur_best = accuracy
        cur_node = token_to_flip
        print(f"token2flip: {token_to_flip}, new candidates {candidates}")
        for c in candidates:
            new_prompt = self.prompt_candidates.clone()
            new_prompt[token_to_flip_idx] = c
            new_prompt = new_prompt.tile(self.data.num_nodes).view(self.data.num_nodes, -1)
            accuracy, logits = self.edge2node_logits('val', new_prompt, emb)
            print(accuracy, cur_best)
            if accuracy[0.5] > cur_best[0.5]:
                cur_best = accuracy
                cur_node = c

        self.prompt_candidates[token_to_flip_idx] = cur_node
        print("#####")
        # TODO: this still does not work; prompt candidates do not change. need fixing
        # print(self.prompt_candidates)
        return {'loss_train': loss.item()}

    def prompt_test(self):
        candidates = self.prompt_candidates.tile(self.data.num_nodes).view(self.data.num_nodes, -1)
        emb = self.node_embeddings.embs
        test_correct, test_logits = self.edge2node_logits('test', candidates, emb)
        val_correct, val_logits = self.edge2node_logits('valid', candidates, emb)
        stats = {}
        for k in test_correct.keys():
            stats[f'test@{k}'] = test_correct[k] / test_logits.size(0)
            stats[f'val@{k}'] = val_correct[k] / val_logits.size(0)
        return stats

    def prompt_stats(self, best_stats, stats, bad_counter):
        if stats['val@0.5'] > best_stats['val@0.5']:
            best_stats.update(stats)
            bad_counter = 0
        else:
            bad_counter += 1
        return best_stats, bad_counter

    def train_test_frame(self, runs_fn, stats_fn):
        best_stats = None
        patience = self.args.patience
        bad_counter = 0.
        # val_loss_history = []
        for epoch in range(self.args.epochs):
            stats = runs_fn()
            best_stats = stats if best_stats is None else best_stats
            best_stats, bad_counter = stats_fn(best_stats, stats, bad_counter)
            if bad_counter == patience:
                break
            if epoch % 20 == 0 or epoch == 1:
                log = f"Epoch: {epoch:03d}, "
                for k, v in best_stats.items():
                    log = log + f"{k}: {v:.4f}, "
                print(log)
        log = ''
        for k, v in best_stats.items():
            log = log + f"{k}: {v:.4f}, "
        print(log)
        return best_stats

    def edge2node_prediction(self):
        stats = {}
        test_correct, test_logits = self.edge2node_logits('test')
        # val_correct, val_logits = self.edge2node_logits('valid')
        train_correct, train_logits = self.edge2node_logits('train')

        for k in test_correct.keys():
            stats[f'test@{k}'] = test_correct[k] / test_logits.size(0)
            # stats[f'val@{k}'] = val_correct[k] / val_logits.size(0)
            stats[f'train@{k}'] = train_correct[k] / train_logits.size(0)
        return stats

    def edge2node_logits(self, mode='test', prompt=None, embs=None):
        # self.model.eval()
        # self.edge_predictor.eval()
        # import pdb; pdb.set_trace()
        if embs is None:
            embs = self.model(self.data.x, self.data.edge_index)
        if prompt is not None:
            embs = self.build_prompt(embs, prompt)
        train_embs = embs[self.data.train_mask]
        train_labels = self.data.y[self.data.train_mask]
        train_counter = collections.Counter(train_labels.cpu().tolist())
        #
        test_embs = embs[self.data.test_mask]
        test_labels = self.data.y[self.data.test_mask]
        #
        val_embs = embs[self.data.val_mask]
        val_labels = self.data.y[self.data.val_mask]

        # TODO : a bit hacky, need to figure out how to improve inference speed.
        def get_correction(test_embs, test_labels):
            correct = {0.25: 0, 0.5: 0, 0.75: 0, 0.9: 0}
            logits = []
            for sample in range(test_embs.size(0)):
                tile_embs = test_embs[sample].tile(train_embs.size(0)).reshape(train_embs.size(0), -1)
                pred_edges = torch.sigmoid(self.edge_predictor(tile_embs, train_embs)).squeeze()
                #
                pr_class = []
                for i in range(max(train_counter.keys()) + 1):
                    p = pred_edges[train_labels.eq(i)].mean().unsqueeze(0)
                    pr_class.append(p)
                logits.append(torch.cat(pr_class, dim=0).unsqueeze(0))
                #
                for threshold in correct.keys():
                    pred_labels = train_labels[pred_edges.gt(threshold)]
                    class_counter = collections.Counter(pred_labels.cpu().tolist())
                    if len(class_counter):
                        max_label = max(class_counter, key=class_counter.get)
                        if max_label == test_labels[sample]:
                            correct[threshold] += 1
            return correct, torch.cat(logits, dim=0)

        if mode == 'test':
            return get_correction(test_embs, test_labels)
        elif mode == 'valid':
            return get_correction(val_embs, val_labels)
        else:
            return get_correction(train_embs, train_labels)

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