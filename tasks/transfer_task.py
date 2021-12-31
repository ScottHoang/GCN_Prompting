import random

from .edge_task import EdgeLearner
from .node_task import NodeLearner
from utils import TaskPredictor
import torch
from utils import CompoundOptimizers
import torch_geometric.transforms as T

class TransNodeWrapper:
    def __init__(self, model, predictor, embeddings, task, prompt_mode, concat_mode, data_features, k_prompt):
        self.model = model
        self.predictor = predictor
        self.prompt_mode = prompt_mode
        self.concat_mode = concat_mode
        self.data_features = data_features
        self.task = task
        self.k_prompt = k_prompt
        self.embeddings = embeddings
        self.prompt = None
        # self.prompts = prompts
        self.is_mlp = isinstance(self.predictor, TaskPredictor)
        if not self.is_mlp:
            self.norm = lambda x: x.div_(x.sum(dim=-1, keepdims=True).clamp(min=1.))

        self.init_optimizer()

    def init_optimizer(self):
        self.optimizer = CompoundOptimizers([self.predictor.optimizer, self.embeddings.optimizer])

    def __call__(self, x, edge_index=None):
        embs = self.embeddings.static_embs
        learnable_embs = self.embeddings.embs
        if self.k_prompt:
            if self.prompt is None:
                self.prompt = prompt = self.get_bfs_prompts(x.size(0), edge_index)
            else:
                prompt = self.prompt
            prompt_embs = self.build_prompt(learnable_embs,  prompt)
        else:
            prompt_embs = None
        embs = self.concat_features(embs, x, prompt_embs)

        if self.is_mlp:
            return self.predictor(embs)
        else:
            embs = self.norm(embs)
            return self.predictor(embs, edge_index)

    def train(self):
        self._cache = True
        self.predictor.train()
        self.model.eval()

    def eval(self):
        self._cache = True
        self._cache_embs = None
        self.model.eval()
        self.predictor.eval()

    def build_prompt(self, embs, prompts_idx=None):
        prompts = embs[prompts_idx]
        pmode = self.prompt_mode
        if pmode == 'concat':
            prompts = prompts.reshape(prompts.size(0), -1)
        elif pmode == 'sum':
            prompts = prompts.sum(dim=1)
        elif pmode == 'mean':
            prompts = prompts.mean(dim=1)
        else:
            raise ValueError
        return prompts

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

    def get_bfs_prompts(self, num_nodes, edge_index):
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

def create_domain_transfer_task(model, predictor, embeddings, task, prompt_mode, concat_mode, k_prompts, data, dataset,
                                batch_size, type_trick, split_idx):
    wrap = TransNodeWrapper(model, predictor, embeddings, task, prompt_mode, concat_mode, data.x, k_prompts)
    return NodeLearner(wrap, batch_size, data, dataset, type_trick, split_idx)
