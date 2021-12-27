import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from .generic_task import Learner

class EdgeModelWrapper:
    def __init__(self, model, predictor):
        self.model = model
        self.predictor = predictor

    def __call__(self, x, edge_index, source, target):
        if not self._cache or self._cache_embs is None:
            self._cache_embs = self.model(x, edge_index)
        return self.predictor(self._cache_embs[source], self._cache_embs[target])

    def optimizers_zero_grad(self):
        self.model.optimizer.zero_grad()
        self.predictor.optimizer.zero_grad()

    def optimizers_step(self):
        self.model.optimizer.step()
        self.predictor.optimizer.step()

    def train(self):
        self._cache = False
        self.model.train()
        self.predictor.train()

    def eval(self):
        self._cache = True
        self._cache_embs = None
        self.model.eval()
        self.predictor.eval()

class EdgeLearner(Learner):
    def __init__(self, model:EdgeModelWrapper, batch_size:int, data):
        super().__init__(model, batch_size, data)
        assert isinstance(model, EdgeModelWrapper)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def task_train(self):
        self.model.train()
        loss_epoch = []
        data = self.data
        device = self.data.x.device
        for perm in DataLoader(range(data.train_pos.size(1)), self.batch_size, shuffle=True):

            pos_prediction = self.model(data.x, data.edge_index, data.train_pos[0, perm], data.train_pos[1, perm])
            neg_prediction = self.model(data.x, data.edge_index, data.train_neg[0, perm], data.train_neg[1, perm])

            loss = self.loss_fn(pos_prediction, torch.ones(pos_prediction.shape).to(device)) + \
                   self.loss_fn(neg_prediction, torch.zeros(neg_prediction.shape).to(device))
            self.model.optimizers_zero_grad()
            loss.backward()
            self.model.optimizers_step()
            loss_epoch.append(loss.item())
        mean_loss = sum(loss_epoch) / len(loss_epoch)
        return {'loss_train': mean_loss}

    @torch.no_grad()
    def task_test(self):
        self.model.eval()
        data = self.data
        val_pred_pos = []
        val_pred_neg = []
        for perm in DataLoader(range(data.val_pos.size(1)), self.batch_size, shuffle=True):
            #  val
            val_pred_pos.append(torch.sigmoid(
                self.model(data.x, data.edge_index, data.val_pos[0, perm], data.val_pos[1, perm])))
            val_pred_neg.append(torch.sigmoid(
                self.model(data.x, data.edge_index, data.val_neg[0, perm], data.val_neg[1, perm])))
        val_pred_pos = torch.cat(val_pred_pos, dim=0)
        val_pred_neg = torch.cat(val_pred_neg, dim=0)
        val_pred = torch.cat([val_pred_pos, val_pred_neg], dim=0).squeeze(-1).cpu()
        val_labels = torch.cat([torch.ones(val_pred_pos.shape), torch.zeros(val_pred_neg.shape)]).squeeze(-1).cpu()
        val_roc_auc = roc_auc_score(val_labels, val_pred)
        val_ap = average_precision_score(val_labels, val_pred)
        #  test
        test_pred_pos = []
        test_pred_neg = []
        for perm in DataLoader(range(data.test_pos.size(1)), self.batch_size, shuffle=False):
            test_pred_pos.append(torch.sigmoid(
                self.model(data.x, data.edge_index, data.test_pos[0, perm], data.test_pos[1, perm])))
            test_pred_neg.append(torch.sigmoid(
                self.model(data.x, data.edge_index, data.test_neg[0, perm], data.test_neg[1, perm])))

        test_pred_pos = torch.cat(test_pred_pos, dim=0)
        test_pred_neg = torch.cat(test_pred_neg, dim=0)
        test_pred = torch.cat([test_pred_pos, test_pred_neg], dim=0).squeeze(-1).cpu()
        test_labels = torch.cat([torch.ones(test_pred_pos.shape), torch.zeros(test_pred_neg.shape)]).squeeze(-1).cpu()
        test_roc_auc = roc_auc_score(test_labels, test_pred)
        test_ap = average_precision_score(test_labels, test_pred)
        return {'valid_ap': val_ap, 'val_roc_auc': val_roc_auc,
                'test_ap': test_ap, 'test_roc_auc': test_roc_auc}

    @staticmethod
    def stats(best_stats, stats, bad_counter):
        if stats['val_roc_auc'] > best_stats['val_roc_auc']:
            best_stats.update(stats)
            bad_counter = 0
        else:
            bad_counter += 1
        return best_stats, bad_counter

def create_edge_task(model, predictor, batch_size, data):
    wrap = EdgeModelWrapper(model, predictor)
    return EdgeLearner(wrap, batch_size, data)
