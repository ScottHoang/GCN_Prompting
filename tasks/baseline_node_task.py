from .node_task import NodeLearner
from .transfer_task import TransNodeWrapper
from utils import CompoundOptimizers
from torch.nn import functional as F

class BaseLineNodeWrapper(TransNodeWrapper):
    def __init__(self, model, predictor):
        super(BaseLineNodeWrapper, self).__init__(model, predictor, None, None, None, None, None, None)

    def init_optimizer(self):
        self.optimizer = CompoundOptimizers([self.model.optimizer,
                                             self.predictor.optimizer])

    def __call__(self, x, edge_index=None):
        embs = self.model(x, edge_index)
        embs =  F.relu(embs)
        if self.is_mlp:
            return self.predictor(embs)
        else:
            return self.predictor(embs, edge_index)


def create_node_task(model, predictor, batch_size, data, dataset, type_trick, split_idx):
    wrap = BaseLineNodeWrapper(model, predictor)
    return NodeLearner(wrap, batch_size, data, dataset, type_trick, split_idx)
