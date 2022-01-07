import collections
import random

import utils
from .edge_task import EdgeLearner
from .node_task import NodeLearner
from utils import TaskPredictor
import torch
from utils import CompoundOptimizers
import torch_geometric.transforms as T
from utils import MAD, shortest_path
from torch.nn.functional import dropout
import torch.nn as nn
class TransNodeWrapper:
    def __init__(self, model, predictor, embeddings, task, prompt_mode, concat_mode, data, k_prompt, prompt_type, prompt_raw,
                 prompt_continual, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model = model
        self.prompt_type = prompt_type
        self.predictor = predictor
        self.prompt_mode = prompt_mode
        self.prompt_continual = prompt_continual
        self.concat_mode = concat_mode
        self.data = data
        self.task = task
        self.k_prompt = k_prompt
        self.embeddings = embeddings
        self.prompt = None
        self.prompt_raw = prompt_raw
        # self.prompts = prompts
        self.is_mlp = isinstance(self.predictor, TaskPredictor)
        if not self.is_mlp:
            self.norm = lambda x: x.div_(x.sum(dim=-1, keepdims=True).clamp(min=1.))

        self.stats = collections.defaultdict(list)
        self.index = self.cos_distance = None

        self.init_optimizer()
        self.training = True


    def init_optimizer(self):
        self.optimizer = CompoundOptimizers([self.predictor.optimizer, self.embeddings.optimizer])

    def __call__(self, x, edge_index=None):
        if self.prompt_raw:
            learnable_embs = self.embeddings.embs_raw
            embs = self.embeddings.embs
            emb_tgt = None
            emb_src = learnable_embs
        else:
            learnable_embs = self.embeddings.embs
            embs = self.embeddings.static_embs
            emb_tgt = learnable_embs
            emb_src = embs
        if self.k_prompt:
            if self.prompt_continual:
                self.prompt = prompt = self.get_prompt(emb_src, emb_tgt, x.size(0), edge_index)
            else:
                if self.prompt is None:
                    self.prompt = prompt = self.get_prompt(emb_src, emb_tgt, x.size(0), edge_index)
                else:
                    prompt = self.prompt

            embs = self.build_prompt(embs, learnable_embs, prompt)
        if self.concat_mode:
            embs = torch.cat([embs, x], dim=-1)

        if self.is_mlp:
            return self.predictor(embs)
        else:
            # embs = self.norm(embs)
            return self.predictor(embs, edge_index)


    def train(self):
        self._cache = True
        self.predictor.train()
        self.model.eval()
        self.training = True

    def eval(self):
        self._cache = True
        self._cache_embs = None
        self.model.eval()
        self.predictor.eval()
        self.training = False

    def build_prompt(self, embs, prompt_embs, prompts_idx=None):
        prompts = prompt_embs[prompts_idx]
        pmode = self.prompt_mode
        if pmode == 'concat':
            prompts = prompts.reshape(prompts.size(0), -1)
        elif pmode == 'sum':
            prompts = prompts.sum(dim=1)
        elif pmode == 'mean':
            prompts = prompts.mean(dim=1)
        else:
            raise ValueError
        embs = torch.cat([embs, prompts], dim=-1)
        return embs

    def concat_features(self, embs, data_features, prompts=None):
        cmode = self.concat_mode
        if cmode:
            if prompts is not None:
                embs = torch.cat([embs, prompts, data_features], dim=-1)
            else:
                embs = torch.cat([embs, data_features], dim=-1)
        else:
            if prompts is not None:
                embs = torch.cat([embs, prompts], dim=-1)
            else:
                pass
        return embs

    def get_prompt(self, x_src, x_tgt, num_nodes, edge_index):
        self.index = getattr(self, f"get_{self.prompt_type}_prompts")(x_src, x_tgt, num_nodes, edge_index)
        # if self.training:
        #     self.analyze_prompt(x_src, x_tgt, self.index)
        return self.index
    
    def analyze_prompt(self, src, tgt, index):
        if not self.prompt_continual or self.prompt_type == 'bfs':
            if self.prompt_type == 'bfs':
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
        assert self.k_prompt > 0
        tgt = torch.ones(num_nodes, num_nodes).to(x_src.device)
        cos_distance = MAD(x_src, tgt, mean=False, emb_tgt=x_tgt)
        self.cos_distance = cos_distance.detach().clone()
        cos_distance *= -1
        cos_distance.fill_diagonal_(float("-inf"))
        return cos_distance.topk(self.k_prompt, dim=-1)[1].squeeze()

    def get_madmin_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        assert self.k_prompt > 0
        tgt = torch.ones(num_nodes, num_nodes).to(x_src.device)
        self.cos_distance = cos_distance = MAD(x_src, tgt, mean=False, emb_tgt=x_tgt)
        return cos_distance.topk(self.k_prompt, dim=-1)[1].squeeze()

    def get_m2d_prompts(self, x_src, x_tgt, num_nodes, edge_index):
        assert hasattr(self, 'distance')
        distance = self.distance
        tgt = torch.ones(num_nodes, num_nodes).to(x_src.device)
        self.cos_distance = cos_distance = MAD(x_src, tgt, mean=False, emb_tgt=x_tgt)
        # cos_distance.fill_diagonal_(-1)
        distance[distance.eq(510)] = -1
        distance[distance.eq(-1)] = distance.max() + 1
        score = distance / cos_distance.sqrt().fill_diagonal_(-1)
        index = score.topk(self.k_prompt, dim=-1)[1].squeeze()
        return index

    def get_bfs_prompts(self, x, x_tgt, num_nodes, edge_index):
        assert self.k_prompt > 0
        target_k = self.k_prompt
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
    

def create_domain_transfer_task(model, predictor, embeddings, task, prompt_mode, concat_mode, k_prompts, data, dataset,
                                batch_size, type_trick, split_idx, prompt_type, prompt_raw, prompt_continual, **kwargs):
    wrap = TransNodeWrapper(model, predictor, embeddings, task, prompt_mode, concat_mode, data, k_prompts, prompt_type, prompt_raw, prompt_continual, **kwargs)
    return NodeLearner(wrap, batch_size, data, dataset, type_trick, split_idx)
