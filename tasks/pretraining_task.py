import math

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

import utils
from .generic_task import Learner


class PretrainWrapper:
    def __init__(self, model, edge_predictor=None, attr_predictor=None):
        self.model = model
        self.edge_predictor = edge_predictor
        self.attr_predictor = attr_predictor
        self.training = True

    def forward_embs(self, x, edge_index):
        return self.model(x, edge_index)

    def forward_edge(self, x, edge_index, source, target):
        assert self.edge_predictor is not None
        embs = self.forward_embs(x, edge_index)
        return self.edge_predictor(embs[source], embs[target])

    def forward_attr(self, masked_x, edge_index):
        assert self.attr_predictor is not None
        embs = self.forward_embs(masked_x, edge_index)
        return self.attr_predictor(embs)

    def optimizers_zero_grad(self):
        self.model.optimizer.zero_grad()
        if self.edge_predictor is not None:
            self.edge_predictor.optimizer.zero_grad()
        if self.attr_predictor is not None:
            self.attr_predictor.optimizer.zero_grad()

    def optimizers_step(self):
        self.model.optimizer.step()
        if self.edge_predictor is not None:
            self.edge_predictor.optimizer.step()
        if self.attr_predictor is not None:
            self.attr_predictor.optimizer.step()

    def train(self):
        self._cache = False
        self.model.train()
        if self.edge_predictor is not None:
            self.edge_predictor.train()
        if self.attr_predictor is not None:
            self.attr_predictor.train()
        self.training = True

    def eval(self):
        self._cache = True
        self._cache_embs = None
        self.model.eval()
        if self.edge_predictor is not None:
            self.edge_predictor.eval()
        if self.attr_predictor is not None:
            self.attr_predictor.eval()
        self.training = False


class PretrainLearner(Learner):
    def __init__(self, model: PretrainWrapper, args, batch_size, data, temp):

        super().__init__(model, batch_size, data)
        for k, v in vars(args).items():
            setattr(self, k, v)
        # assert isinstance(model, PretrainWrapper)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn2 = torch.nn.CrossEntropyLoss()
        self.temp = 1.0

    def task_train(self):
        self.model.train()
        loss_epoch = []
        loss = []
        data = self.data
        # masked_edges, train_pos_edge_index, train_neg_edge_index = sampling_edges(data.train_pos_edge_index, data.train_neg_adj_mask)

        if utils.AcontainsB(self.prompt_pretrain_type, ['edgeMask']):
            pos_prediction = self.model.forward_edge(data.x, data.edge_index, data.train_pos_edge_index[0],
                                                     data.train_pos_edge_index[1])
            loss_edges = self.loss_fn(pos_prediction, torch.ones_like(pos_prediction))
            loss.append(loss_edges)
        if utils.AcontainsB(self.prompt_pretrain_type, ['contrastive']):
            embs = self.model.forward_embs(data.x, data.train_pos_edge_index)
            logits, labels = self.info_nce_loss(embs, data.train_pos_edge_index[0], data.train_pos_edge_index[1],
                                                data.train_neg_edge_index[0], data.train_neg_edge_index[1], self.temp)
            loss.append(self.loss_fn(logits, labels))
        if utils.AcontainsB(self.prompt_pretrain_type, ['attrMask']):
            # x, masked_idx = sampling_attr(data.x)
            pca = self.model.forward_attr(self.data.x, data.edge_index)
            loss.append(F.mse_loss(pca[self.data.train_mask], data.pca[self.data.train_mask]))
        loss = sum(loss) / len(loss)
        self.model.optimizers_zero_grad()
        loss.backward()
        self.model.optimizers_step()
        loss_epoch.append(loss.item())
        mean_loss = sum(loss_epoch) / len(loss_epoch)
        return {'loss_train': mean_loss}

    @torch.no_grad()
    def task_test(self):
        self.model.eval()
        results = {}
        if utils.AcontainsB(self.prompt_pretrain_type, ['edgeMask']):
            results.update(self.task_test_edges())
        if utils.AcontainsB(self.prompt_pretrain_type, ['contrastive']):
            results.update(self.task_test_contrastive())
        if utils.AcontainsB(self.prompt_pretrain_type, ['attrMask']):
            results.update(self.task_test_attrMask())

        return results

    def task_test_contrastive(self):
        data = self.data
        embs_valid = self.model.forward_embs(data.x, data.val_pos_edge_index)
        embs_test = self.model.forward_embs(data.x, data.test_pos_edge_index)

        logits, labels = self.info_nce_loss(embs_valid, data.val_pos_edge_index[0], data.val_pos_edge_index[1],
                                            data.val_neg_edge_index[0],
                                            data.val_neg_edge_index[1], self.temp)
        loss_valid = self.loss_fn(logits, labels)
        logits, labels = self.info_nce_loss(embs_test, data.test_pos_edge_index[0], data.test_pos_edge_index[1],
                                            data.test_neg_edge_index[0],
                                            data.test_neg_edge_index[1], self.temp)
        loss_test = self.loss_fn(logits, labels)
        return {"loss_vcl": loss_valid.item(), 'loss_tcl': loss_test.item()}

    def task_test_attrMask(self):
        data = self.data
        pca = self.model.forward_attr(data.x, data.edge_index)
        l_pca_valid = F.mse_loss(pca[data.val_mask], data.pca[data.val_mask])
        l_pca_test = F.mse_loss(pca[data.test_mask], data.pca[data.test_mask])
        return {'loss_vpca': l_pca_valid.item(), 'loss_tpca': l_pca_test.item()}

    def task_test_edges(self):
        data = self.data
        val_pred_pos = []
        val_pred_neg = []
        test_pred_pos = []
        test_pred_neg = []
        val_pred_pos.append(torch.sigmoid(
            self.model.forward_edge(data.x, data.val_pos_edge_index, data.val_pos_edge_index[0],
                                    data.val_pos_edge_index[1])))
        val_pred_neg.append(torch.sigmoid(
            self.model.forward_edge(data.x, data.val_pos_edge_index, data.val_neg_edge_index[0],
                                    data.val_neg_edge_index[1])))
        test_pred_pos.append(torch.sigmoid(
            self.model.forward_edge(data.x, data.test_pos_edge_index, data.test_pos_edge_index[0],
                                    data.test_pos_edge_index[1])))
        test_pred_neg.append(torch.sigmoid(
            self.model.forward_edge(data.x, data.test_pos_edge_index, data.test_neg_edge_index[0],
                                    data.test_neg_edge_index[1])))

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

    def stats(self, best_stats, stats, bad_counter):
        if utils.AcontainsB(self.prompt_pretrain_type, ['edgeMask']):
            if stats['val_roc_auc'] > best_stats['val_roc_auc']:
                best_stats.update(stats)
                bad_counter = 0
            else:
                bad_counter += 1
        elif utils.AcontainsB(self.prompt_pretrain_type, ['contrastive']):
            if stats['loss_vcl'] < best_stats['loss_vcl']:
                best_stats = stats
            else:
                bad_counter += 1
        else:
            if stats['loss_vpca'] < best_stats['loss_vpca']:
                best_stats = stats
            else:
                bad_counter += 1
        return best_stats, bad_counter


def neg_sampling(neg_adj_mask, ratio, row):
    n_t = int(math.floor(ratio * row.size(0)))
    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]
    return neg_row, neg_col


def sampling_edges(edge_index, neg_adj_mask, ratio=0.2):
    pos_row, pos_col = edge_index
    num_mask = int(math.floor(ratio * pos_row.size(0)))
    mask_vector = torch.ones(pos_row.size(0))
    mask_idx = torch.randperm(pos_row.size(0))[0:num_mask]
    #
    mask_vector[mask_idx] = 0
    masked_edge = edge_index[:, mask_vector.bool()]
    #
    pos_row, pos_col = pos_row[mask_idx], pos_col[mask_idx]
    neg_row, neg_col = neg_sampling(neg_adj_mask, ratio, pos_row)
    return masked_edge, [pos_row, pos_col], [neg_row, neg_col]


def sampling_attr(x, ratio=0.2):
    x = x.clone()
    N, attr_size = x.shape
    num_mask = int(math.floor(ratio * N))
    mask_idx = torch.randperm(N)[0:num_mask]
    x[mask_idx] = 0
    return x, mask_idx


def create_pretrain_task(model, batch_size, data, args, temp=1.0, edge_predictor=None, attr_predictor=None):
    wrap = PretrainWrapper(model, edge_predictor, attr_predictor)
    return PretrainLearner(wrap, args, batch_size, data, temp)
