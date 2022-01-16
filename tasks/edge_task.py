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
        self.training = True

    def __call__(self, x, edge_index, source, target):
        if not self._cache or self._cache_embs is None:
            self._cache_embs = self.model(x, edge_index)
        if self.training:
            return self.predictor(self._cache_embs[source], self._cache_embs[target]), self._cache_embs
        else:
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
        self.training = True

    def eval(self):
        self._cache = True
        self._cache_embs = None
        self.model.eval()
        self.predictor.eval()
        self.training = False


class EdgeLearner(Learner):
    def __init__(self, model: EdgeModelWrapper, batch_size: int, data, temp):
        super().__init__(model, batch_size, data)
        assert isinstance(model, EdgeModelWrapper)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn2 = torch.nn.CrossEntropyLoss()
        self.temp = 1.0

    def task_train(self):
        self.model.train()
        loss_epoch = []
        data = self.data
        device = self.data.x.device
        if self.batch_size > 0:
            for perm in DataLoader(range(data.train_pos.size(1)), self.batch_size, shuffle=True):
                pos_prediction,  embs = self.model(data.x, data.train_pos, data.train_pos[0, perm], data.train_pos[1, perm])
                # neg_prediction, _ = self.model(data.x, data.edge_index, data.train_neg[0, perm], data.train_neg[1, perm])

                loss = self.loss_fn(pos_prediction, torch.ones(pos_prediction.shape).to(device))
                # + \
                       # self.loss_fn(neg_prediction, torch.zeros(neg_prediction.shape).to(device))
                # logits, labels = self.info_nce_loss(embs, data.train_pos[0], data.train_pos[1], data.train_neg[0],
                #                                     data.train_neg[1], self.temp)
                # loss = loss + self.loss_fn(logits, labels)
                self.model.optimizers_zero_grad()
                loss.backward()
                self.model.optimizers_step()
                loss_epoch.append(loss.item())
        else:
            pos_prediction, embs = self.model(data.x, data.train_pos, data.train_pos[0], data.train_pos[1])
            # neg_prediction, _ = self.model(data.x, data.edge_index, data.train_neg[0], data.train_neg[1])

            loss = self.loss_fn(pos_prediction, torch.ones(pos_prediction.shape).to(device))
            # + \
                   # self.loss_fn(neg_prediction, torch.zeros(neg_prediction.shape).to(device))
            # logits, labels = self.info_nce_loss(embs, data.train_pos[0], data.train_pos[1], data.train_neg[0], data.train_neg[1], self.temp)
            # loss = loss + self.loss_fn(logits, labels)
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
        test_pred_pos = []
        test_pred_neg = []
        if self.batch_size > 0:
            for perm in DataLoader(range(data.val_pos.size(1)), self.batch_size, shuffle=True):
                #  val
                val_pred_pos.append(torch.sigmoid(
                    self.model(data.x, data.edge_index, data.val_pos[0, perm], data.val_pos[1, perm])))
                val_pred_neg.append(torch.sigmoid(
                    self.model(data.x, data.edge_index, data.val_neg[0, perm], data.val_neg[1, perm])))

            for perm in DataLoader(range(data.test_pos.size(1)), self.batch_size, shuffle=False):
                test_pred_pos.append(torch.sigmoid(
                    self.model(data.x, data.edge_index, data.test_pos[0, perm], data.test_pos[1, perm])))
                test_pred_neg.append(torch.sigmoid(
                    self.model(data.x, data.edge_index, data.test_neg[0, perm], data.test_neg[1, perm])))
        else:
            val_pred_pos.append(torch.sigmoid(
                self.model(data.x, data.edge_index, data.val_pos[0], data.val_pos[1])))
            val_pred_neg.append(torch.sigmoid(
                self.model(data.x, data.edge_index, data.val_neg[0], data.val_neg[1])))
            test_pred_pos.append(torch.sigmoid(
                self.model(data.x, data.edge_index, data.test_pos[0], data.test_pos[1])))
            test_pred_neg.append(torch.sigmoid(
                self.model(data.x, data.edge_index, data.test_neg[0], data.test_neg[1])))

        val_pred_pos = torch.cat(val_pred_pos, dim=0)
        val_pred_neg = torch.cat(val_pred_neg, dim=0)
        val_pred = torch.cat([val_pred_pos, val_pred_neg], dim=0).squeeze(-1).cpu()
        val_labels = torch.cat([torch.ones(val_pred_pos.shape), torch.zeros(val_pred_neg.shape)]).squeeze(-1).cpu()
        val_roc_auc = roc_auc_score(val_labels, val_pred)
        val_ap = average_precision_score(val_labels, val_pred)
        #  test

        test_pred_pos = torch.cat(test_pred_pos, dim=0)
        test_pred_neg = torch.cat(test_pred_neg, dim=0)
        test_pred = torch.cat([test_pred_pos, test_pred_neg], dim=0).squeeze(-1).cpu()
        test_labels = torch.cat([torch.ones(test_pred_pos.shape), torch.zeros(test_pred_neg.shape)]).squeeze(-1).cpu()
        test_roc_auc = roc_auc_score(test_labels, test_pred)
        test_ap = average_precision_score(test_labels, test_pred)
        return {'valid_ap': val_ap, 'val_roc_auc': val_roc_auc,
                'test_ap': test_ap, 'test_roc_auc': test_roc_auc}

    def info_nce_loss(self, features, pos_src, pos_tgt, neg_src, neg_tgt, temp=1.0):

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T).fill_diagonal_(0)

        positives = similarity_matrix[pos_src, pos_tgt]

        negatives = similarity_matrix[neg_src, neg_tgt]

        logits = torch.cat([positives, negatives], dim=0)
        logits = logits / temp

        labels = torch.cat([torch.ones_like(positives), torch.zeros_like(negatives)], dim=0)
        return logits, labels

    @staticmethod
    def stats(best_stats, stats, bad_counter):
        if stats['val_roc_auc'] > best_stats['val_roc_auc']:
            best_stats.update(stats)
            bad_counter = 0
        else:
            bad_counter += 1
        return best_stats, bad_counter


def create_edge_task(model, predictor, batch_size, data, temp=1.0):
    wrap = EdgeModelWrapper(model, predictor)
    return EdgeLearner(wrap, batch_size, data, temp)

