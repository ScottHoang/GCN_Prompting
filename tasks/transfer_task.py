import collections
import datetime
import os
import random

import torch_geometric

import utils as U
from .edge_task import EdgeLearner
from .node_task import NodeLearner
from utils import TaskPredictor
import torch
from utils import CompoundOptimizers
import torch_geometric.transforms as T
from utils import MAD, shortest_path
from torch.nn.functional import dropout
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as TGU
class Att(nn.Module):
    def __init__(self, num_nodes, in_c, nhead=8, num_encoder=6, batch_first=True, dropout=0.5, lr=1e-3):
        super(Att, self).__init__()
        layer = nn.TransformerEncoderLayer(in_c, nhead, dropout=dropout, batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(layer, num_encoder)
        self.pe = nn.Parameter(torch.empty(num_nodes, 1, in_c), requires_grad=True)
        self.init_weights()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=[0.9, 0.98])

    def init_weights(self):
        nn.init.normal_(self.pe)

    def forward(self, x):
        x = x + self.pe
        output = self.encoder(x)
        return output.mean(dim=1)


class TransNodeWrapper:
    def __init__(self, model, predictor, embeddings, args, task, data):
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.model = model
        self.predictor = predictor
        self.data = data
        self.task = task
        self.embeddings = embeddings
        self.prompt = None
        self.is_mlp = isinstance(self.predictor, TaskPredictor)
        if not self.is_mlp:
            self.norm = lambda x: x.div_(x.sum(dim=-1, keepdims=True).clamp(min=1.))

        self.stats = collections.defaultdict(list)
        self.index = self.cos_distance = None

        if self.prompt_aggr == 'att':
            in_c = self.embeddings.embs.size(1)
            self.att_module = Att(self.data.x.size(0), in_c, self.att_head, self.att_num_layer, self.att_dropout, self.att_lr)
            self.att_module.to(self.data.x.device)

        self.init_optimizer()
        self.training = True

    def init_optimizer(self):
        if self.prompt_aggr == 'att':
            self.optimizer = CompoundOptimizers([self.predictor.optimizer, self.embeddings.optimizer,
                                                 self.att_module.optimizer], [self.embeddings.scheduler])
        else:
            schedulers = [self.embeddings.scheduler]
            if hasattr(self.predictor, 'scheduler'):
                schedulers.append(self.predictor.scheduler)
            self.optimizer = CompoundOptimizers([self.predictor.optimizer, self.embeddings.optimizer],
                                                schedulers)

    def __call__(self, x, edge_index=None):
        if self.prompt_raw:
            prompt_embs = self.embeddings.embs_raw
            self.node_embs = node_embs = self.embeddings.embs
            emb_tgt = None
            emb_src = prompt_embs
        else:
            self.node_embs = emb_src = emb_tgt = self.embeddings.embs
        ############################################
        self.final_embs, self.prompted_embs, x, edge_index = self.get_embs_edge_index(emb_src, emb_tgt, x, edge_index)
        if self.is_mlp:
            return self.predictor(self.final_embs)
        else:
            return self.predictor(self.final_embs, edge_index)

    def get_embs_edge_index(self, emb_src, emb_tgt, x, edge_index):
        if self.prompt_k:
            if self.prompt_continual:
                self.prompt = prompt = self.get_prompt(emb_src, emb_tgt, x.size(0), edge_index)
            else:
                if self.prompt is None:
                    self.prompt = prompt = self.get_prompt(emb_src, emb_tgt, x.size(0), edge_index)
                else:
                    prompt = self.prompt
            if self.training and self.plot_info:
                self.analyze_prompt(emb_src, emb_tgt, prompt)

            prompted_embs, edge_index = self.build_prompt(emb_src, emb_tgt, prompt, edge_index)
        else:
            prompted_embs = emb_src
        if self.prompt_w_org_features:
            final_embs = torch.cat([prompted_embs,x], dim=-1)
        else:
            final_embs = prompted_embs

        return final_embs, prompted_embs, x, edge_index

    def train(self):
        self._cache = True
        self.predictor.train()
        self.model.eval()
        self.training = True
        if hasattr(self, 'att_module'):
            self.att_module.train()

    def eval(self):
        self._cache = True
        self._cache_embs = None
        self.model.eval()
        self.predictor.eval()
        self.training = False
        if hasattr(self, 'att_module'):
            self.att_module.eval()

    def build_prompt(self, embs, prompt_embs, prompts_idx=None, edge_index=None):
        prompts = prompt_embs[prompts_idx]
        pmode = self.prompt_aggr
        if pmode in ['concat', 'sum', 'mean']:
            if pmode == 'concat':
                prompts = prompts.reshape(prompts.size(0), -1)
            elif pmode == 'sum':
                prompts = prompts.sum(dim=1)
            elif pmode == 'mean':
                prompts = prompts.mean(dim=1)
            embs = torch.cat([embs, prompts], dim=-1)
        elif pmode == 'att':
            embs = torch.cat([embs.unsqueeze(dim=1), prompts], dim=1)
            embs = self.att_module(embs)
        elif pmode == 'edges':
            assert edge_index is not None
            src = torch.arange(prompts_idx.size(0)).tile(dims=(self.prompt_k,1)).t().reshape(1, -1).to(edge_index.device)
            tgt = prompts_idx.reshape(1, -1)
            prompt_edge_index = torch.cat([src, tgt], dim=0).to(edge_index.device)
            edge_index = torch.cat([edge_index, prompt_edge_index], dim=-1)
            edge_index = TGU.coalesce(edge_index)
        if edge_index is not None:
            return embs, edge_index
        else:
            return embs

    def get_prompt(self, x_src, x_tgt, num_nodes, edge_index):
        self.index = getattr(self, f"get_{self.prompt_type}_prompts")(x_src, x_tgt, num_nodes, edge_index)
        return self.index
    
    def analyze_prompt(self, src, tgt, index):
        if not self.prompt_continual or self.prompt_type in  ['bfs', 'random', 'ntxent', 'ntxent2',
                                                              'ntxent3', 'exp', 'micmap', 'macmip', 'micmip', 'rubberband']:
            if self.prompt_type in ['bfs', 'random', 'ntxent', 'ntxent2', 'ntxent3', 'exp', 'expv2', 'micmap', 'macmip', 'micmip', 'rubberband']:
                self.cos_distance = MAD(src, torch.ones(src.size(0), src.size(0)).to(src.device),
                                        mean=False, emb_tgt=tgt)
            else:
                self.get_prompt(src, tgt, src.size(0), self.data.edge_index)
        tgt = src if tgt is None else tgt
        mean_distances = []
        mean_i2nr = []
        for i in range(index.size(0)):
            tgt = index[i]
            mean_distances.append(self.cos_distance[i, tgt].mean())
            labels = self.data.y[tgt]
            like_label = labels.eq(self.data.y[i]).sum()
            mean_i2nr.append(like_label.div(tgt.size(0)))
        self.stats['mean_distance'].append(sum(mean_distances).div(len(mean_distances)).item())
        self.stats['mean_i2nr'].append(sum(mean_i2nr).div(len(mean_i2nr)).item())

    def get_madmax_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        assert self.prompt_k > 0
        tgt = torch.ones(num_nodes, num_nodes).to(x_src.device)
        cos_distance = MAD(x_src, tgt, mean=False, emb_tgt=x_tgt)
        self.cos_distance = cos_distance.detach().clone()
        cos_distance *= -1
        cos_distance.fill_diagonal_(float("-inf"))
        return cos_distance.topk(self.prompt_k, dim=-1)[1].squeeze()

    def get_madmin_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        assert self.prompt_k > 0
        tgt = torch.ones(num_nodes, num_nodes).to(x_src.device)
        self.cos_distance = cos_distance = MAD(x_src, tgt, mean=False, emb_tgt=x_tgt)
        return cos_distance.topk(self.prompt_k, dim=-1)[1].squeeze()

    def get_m2d_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        assert hasattr(self, 'distance')
        distance = self.distance
        tgt = torch.ones(num_nodes, num_nodes).to(x_src.device)
        self.cos_distance = cos_distance = MAD(x_src, tgt, mean=False, emb_tgt=x_tgt)
        # cos_distance.fill_diagonal_(-1)
        distance[distance.eq(510)] = -1
        distance[distance.eq(-1)] = distance.max() + 1
        score = distance / cos_distance.sqrt().fill_diagonal_(-1)
        index = score.topk(self.prompt_k, dim=-1)[1]
        return index

    def get_random_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        pop = [i for i in range(x_src.size(0))]
        index = [random.sample(pop, self.prompt_k) for i in range(x_src.size(0))]
        return torch.tensor(index)

    def get_ntxent_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        distance = self.distance
        distance[distance.eq(510)] = 1
        temp = self.prompt_temp
        x_tgt = x_tgt if x_tgt is not None else x_src
        sim_num = torch.exp(U.pair_cosine_similarity(x_src, x_tgt).div(temp))
        sim_denom = sim_num.clone().fill_diagonal_(0).sum(dim=1)
        score = torch.log(sim_num.div(sim_denom.clamp(1e-8)).mul(distance))
        index = score.topk(self.prompt_k, dim=-1)[1].squeeze()
        return index

    def get_ntxent2_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        adj = torch_geometric.utils.to_dense_adj(edge_index).squeeze(0)
        adj.fill_diagonal_(0)
        temp = self.prompt_temp
        x_tgt = x_tgt if x_tgt is not None else x_src
        sim_num = torch.exp(U.pair_cosine_similarity(x_src, x_tgt).div(temp))
        sim_denom = sim_num.clone().mul(adj).sum(dim=1)
        sim_denom[sim_denom.eq(0)] = 1
        score = torch.log(sim_num.div(sim_denom))
        index = score.topk(self.prompt_k, dim=-1)[1].squeeze()
        return index

    def get_ntxent3_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        distance = self.distance
        distance[distance.eq(510)] = 1
        adj = torch_geometric.utils.to_dense_adj(edge_index).squeeze(0)
        adj.fill_diagonal_(0)
        temp = self.prompt_temp
        x_tgt = x_tgt if x_tgt is not None else x_src
        sim_num = torch.exp(U.pair_cosine_similarity(x_src, x_tgt).div(temp))
        sim_denom = sim_num.clone().mul(adj).sum(dim=1)
        sim_denom[sim_denom.eq(0)] = 1
        score = torch.log(sim_num.div(sim_denom).mul(distance))
        index = score.topk(self.prompt_k, dim=-1)[1]
        return index

    def get_exp_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        distance = self.distance.clone()
        distance[distance.eq(510)] = 1
        distance = distance + 1
        # if self.prompt_neighbor_cutoff > 0:
        #     distance[distance.eq(self.prompt_neighbor_cutoff)] = 1
        distance_row_norm = F.normalize(distance.float(), dim=-1)
        distance_col_norm = F.normalize(distance.float(), dim=0)
        distance = (distance_col_norm + distance_row_norm) / 2
        #######
        adj = torch_geometric.utils.to_dense_adj(edge_index).squeeze(0)
        adj.fill_diagonal_(0)
        #####
        sim = U.pair_cosine_similarity(x_src, x_tgt)
        mean_sim = sim.mul(adj).sum(dim=-1).div(adj.sum(dim=-1).clamp(1e-8))
        ######
        score = torch.sigmoid(torch.exp(torch.abs(sim - mean_sim).div(self.prompt_temp)) * distance)
        index = score.topk(self.prompt_k, dim=-1)[1]
        return index

    def get_micmap_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        log_distance, sim, mean_sim = self._prep_micmapmacmip(x_src, x_tgt, num_nodes, edge_index)
        #d#####
        score = torch.exp(sim.sub(mean_sim).div(self.prompt_temp).mul(-1)) * log_distance
        index = score.topk(self.prompt_k, dim=-1)[1]
        return index

    def get_macmip_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        log_distance, sim, mean_sim = self._prep_micmapmacmip(x_src, x_tgt, num_nodes, edge_index)
        #d#####
        score = torch.exp(sim.sub(mean_sim).div(self.prompt_temp)).div(log_distance)
        index = score.topk(self.prompt_k, dim=-1)[1]
        return index

    def get_micmip_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        log_distance, sim, mean_sim = self._prep_micmapmacmip(x_src, x_tgt, num_nodes, edge_index)
        #d#####
        score = torch.exp(sim.sub(mean_sim).div(self.prompt_temp).mul(-1)).div(log_distance)
        index = score.topk(self.prompt_k, dim=-1)[1]
        return index

    def get_rubberband_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        log_distance, sim, mean_sim = self._prep_micmapmacmip(x_src, x_tgt, num_nodes, edge_index)
        #d#####
        score1 = torch.exp(sim.sub(mean_sim).div(self.prompt_temp).mul(-1)) * log_distance
        score2 = torch.exp(sim.sub(mean_sim).div(self.prompt_temp)).div(log_distance)
        score = score1 + score2
        index = score.topk(self.prompt_k, dim=-1)[1]
        return index

    def _prep_micmapmacmip(self, x_src, x_tgt, num_nodes, edge_index):
        distance = self.distance.clone()
        if self.prompt_neighbor_cutoff > 0:
            distance[distance.gt(self.prompt_neighbor_cutoff)] = 1
        else:
            distance[distance.eq(510)] = 1
        distance = distance + 1  # avoid log(1) = 0, instead log(2) = 0.69 is good for punishing node outside of desired range, while not excluding them completely.
        distance.fill_diagonal_(1)  # log(diag) = 0
        distance = (distance + distance.t()) / 2
        log_distance = torch.log(distance / self.prompt_distance_temp)
        #######
        adj = torch_geometric.utils.to_dense_adj(edge_index).squeeze(0)
        adj.fill_diagonal_(0)
        #####
        sim = 1 - U.pair_cosine_similarity(x_src, x_tgt)  # cos distance ~ [0, 2]
        mean_sim = sim.mul(adj).sum(dim=-1).div(adj.sum(dim=-1).clamp(1e-8))
        return log_distance, sim ,mean_sim

    def get_bfs_prompts(self, x, x_tgt, num_nodes, edge_index):
        assert self.prompt_k > 0
        target_k = self.prompt_k
        all_prompts = []
        for i in range(num_nodes):
            prompt = self.bfs(i, target_k, edge_index)
            prompt.sort()
            # prompts.append(i)
            prompt = torch.tensor(prompt)
            if prompt.size(0) < target_k:
                prompt = prompt.tile(target_k // prompt.size(0) + 2)[:target_k]
            all_prompts.append(prompt.unsqueeze(0))
        all_prompts = torch.cat(all_prompts, dim=0)
        return all_prompts

    def bfs(self, node, target, edge_index):
        prompt=set([])
        queue=[node]
        seen = set()

        while queue:
            s = queue.pop(0)
            if s not in seen:
                neighbors = edge_index[1, edge_index[0, :].eq(s)].tolist()
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

    def record_stats(self, stats):
        for k ,v in stats:
            self.stats[k].append(v)

    # def save_embs(self, node_embs, final_embs, prompts):
    #     if self.count % 20 == 0:
    #         torch.save({'node_embs': node_embs, 'final_embs': final_embs, 'prompts': prompts, 'labels': self.data.y,
    #                     'edges': self.data.edge_index, 'lr': self.embeddings.optimizer.param_groups[0]['lr']},
    #                    os.path.join(self.root, f'embeddings_{self.count}.pth.tar'))
    #     self.count += 1

def create_domain_transfer_task(model, predictor, embeddings, args, task, data, split_idx):
    wrap = TransNodeWrapper(model, predictor, embeddings, args, task, data)
    return NodeLearner(args, wrap, args.batch_size, data, args.dataset, args.type_trick, split_idx)
