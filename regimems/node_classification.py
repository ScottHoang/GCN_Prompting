from typing import Dict

from .base_regimen import Regimen
from utils import AcontainsB, evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeClassification(Regimen):
    def __init__(self, args, model, predictor, data, loss_fn, optimizer):
        super().__init__()
        self.init(args)
        self.model = model
        self.predictor = predictor
        self.data = data
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self) -> Dict:
        self.model.train()
        if self.dataset == 'ogbn-arxiv':
            pred = self.model(self.data.x, self.data.edge_index)
            pred = F.log_softmax(pred[self.train_idx], 1)
            loss = self.loss_fn(pred, self.data.y.squeeze(1)[self.train_idx])
        else:
            raw_logits = self.model(self.data.x, self.data.edge_index)
            logits = F.log_softmax(raw_logits[self.data.train_mask], 1)
            loss = self.loss_fn(logits, self.data.y[self.data.train_mask])
            # label smoothing loss
            if AcontainsB(self.type_trick, ['LabelSmoothing']):
                smooth_loss = -raw_logits[self.data.train_mask].mean(dim=-1).mean()
                loss = 0.97 * loss + 0.03 * smooth_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def test(self) -> Dict:
        self.model.eval()
        # torch.cuda.empty_cache()
        if self.dataset == 'ogbn-arxiv':
            out = self.model(self.data.x, self.data.edge_index)
            out = F.log_softmax(out, 1)
            y_pred = out.argmax(dim=-1, keepdim=True)

            train_acc = self.evaluator.eval({
                'y_true': self.data.y[self.split_idx['train']],
                'y_pred': y_pred[self.split_idx['train']],
            })['acc']
            valid_acc = self.evaluator.eval({
                'y_true': self.data.y[self.split_idx['valid']],
                'y_pred': y_pred[self.split_idx['valid']],
            })['acc']
            test_acc = self.evaluator.eval({
                'y_true': self.data.y[self.split_idx['test']],
                'y_pred': y_pred[self.split_idx['test']],
            })['acc']

            return {'train_acc': train_acc, 'valid_acc': valid_acc, 'test_acc': test_acc, "val_los": 0}
        else:
            logits = self.model(self.data.x, self.data.edge_index)
            logits = F.log_softmax(logits, 1)
            acc_train = evaluate(logits, self.data.y, self.data.train_mask)
            acc_val = evaluate(logits, self.data.y, self.data.val_mask)
            acc_test = evaluate(logits, self.data.y, self.data.test_mask)
            val_loss = self.loss_fn(logits[self.data.val_mask], self.data.y[self.data.val_mask])
            return {'train_acc': acc_train, 'valid_acc': acc_val,
                    'test_acc': acc_test, "val_los": val_loss.item()}
