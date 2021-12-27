import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from .generic_task import Learner
from utils import AcontainsB

from ogb.nodeproppred import Evaluator

def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()

class NodeLearner(Learner):
    def __init__(self, model, batch_size: int, data, dataset, type_trick, split_idx=None):
        super().__init__(model, batch_size, data)
        self.loss_fn = F.nll_loss
        self.split_idx = split_idx
        self.dataset = dataset
        self.type_trick = type_trick
        if dataset =='ogbn':
            self.evaluator = Evaluator(name='ogbn-arxiv')
            self.train_idx = self.split_idx['train'].to(self.data.x.device)
        else:
            self.evaluator = None

    def task_train(self):
        self.model.train()
        loss = 0.
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

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        return {'train_loss': loss.item()}

    def task_test(self):
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

    def stats(self, best_stats, stats, bad_counter):
        if self.dataset != 'ogbn-arxiv':
            if best_stats is None or stats['val_los'] < best_stats['val_los']:
                best_stats.update(stats)
                bad_counter = 0
            else:
                bad_counter += 1
        else:
            if stats['valid_acc'] > best_stats['valid_acc']:
                best_stats.update(stats)
                bad_counter = 0
            else:
                bad_counter += 1
        return best_stats, bad_counter


def create_node_task(model, batch_size, data, dataset, type_trick, split_idx=None):
    return NodeLearner(model, batch_size, data, dataset, type_trick, split_idx)