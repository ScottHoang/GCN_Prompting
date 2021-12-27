from .edge_task import EdgeLearner
from .node_task import NodeLearner
from utils import TaskPredictor
import torch
from utils import CompoundOptimizers
class TransNodeWrapper:
    def __init__(self, model, predictor, embeddings, task, prompt_mode, concat_mode, data_features, prompts):
        self.model = model
        self.predictor = predictor
        self.prompt_mode = prompt_mode
        self.concat_mode = concat_mode
        self.data_features = data_features
        self.task = task
        self.embeddings = embeddings
        self.prompts = prompts
        self.is_mlp = isinstance(self.predictor, TaskPredictor)
        self.init_optimizer()

    def init_optimizer(self):
        self.optimizer = CompoundOptimizers([self.predictor.optimizer, self.embeddings.optimizer])

    def __call__(self, x, edge_index=None):
        embs = self.embeddings.static_embs
        learnable_embs = self.embeddings.embs
        if self.prompts is not None:
            prompt_embs = self.build_prompt(learnable_embs,  self.prompts, self.task)
            embs = self.combine_prompt(prompt_embs, embs)

        if self.is_mlp:
            return self.predictor(embs)
        else:
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

    @staticmethod
    def build_prompt(embs, prompt=None, task='dt'):
        # embs = embs / embs.sum(1, keepdims=True).clamp(min=1)
        # embs = embs.detach().clone()
        if task in ['dt']:
            return embs
        elif task == 'dtbfs':
            # prompt = self.bfs()
            assert prompt is not None
            return embs[prompt]
        elif task == 'prompt':
            assert prompt is not None
            _prompt = torch.zeros(prompt.size(0), prompt.size(1)+1, dtype=torch.long)
            _prompt[:, 0] = torch.arange(prompt.size(0))
            _prompt[:, 1::] = prompt
            prompt = _prompt
            return embs[prompt]
        else:
            assert NotImplementedError

    def combine_prompt(self, prompts, embs):
        pmode = self.prompt_mode
        cmode = self.concat_mode
        
        if pmode == 'concat':
            prompts = prompts.reshape(prompts.size(0), -1)
        elif pmode == 'sum':
            prompts = prompts.sum(dim=1)
        elif pmode == 'mean':
            prompts = prompts.mean(dim=1)
        else:
            raise ValueError

        if cmode:
            embs = torch.cat([embs, prompts, self.data_features], dim=-1)
        else:
            embs = torch.cat([embs, prompts], dim=-1)
        return embs

def create_domain_transfer_task(model, predictor, embeddings, task, prompt_mode, concat_mode, prompts, data, dataset,
                                batch_size, type_trick, split_idx):
    if task == 'dt':
        prompts = None
    wrap = TransNodeWrapper(model, predictor, embeddings, task, prompt_mode, concat_mode, data.x, prompts)
    return NodeLearner(wrap, batch_size, data, dataset, type_trick, split_idx)
