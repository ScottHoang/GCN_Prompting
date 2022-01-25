import random

import torch.nn
import torch_geometric.utils

from .generic_task import Learner
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from .transfer_task import TransNodeWrapper
from .node_task import NodeLearner

class VGAELearner(Learner):
    def __init__(self, model, batch_size, data, temp=1.0):
        super(VGAELearner, self).__init__(model, batch_size, data)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.temp = 1.0

    def task_train(self):
        data = self.data
        self.model.train()
        adj = self.get_target_adj(self.data.train_pos_edge_index)
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        pred, mu, logvar = self.model(self.data.x, self.data.train_pos_edge_index)
        loss = loss_function(pred, adj, mu, logvar, self.data.x.size(0), norm ,pos_weight)

        # logits, labels = self.info_nce_loss(mu, data.train_pos_edge_index[0], data.train_pos_edge_index[1], data.train_neg_edge_index[0],  data.train_neg_edge_index[1], self.temp)
        # loss = loss + self.loss_fn(logits, labels)

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        return {'loss_train': loss.item()}

    def get_target_adj(self, edge_index):
        adj = to_dense_adj(edge_index).squeeze(0)
        padding = self.data.size(0) - adj.size(0)
        if padding:
            adj = F.pad(adj, (0, padding, 0, padding))
        return adj

    @torch.no_grad()
    def task_test(self):
        self.model.eval()
        pred, mu, _ = self.model(self.data.x, self.data.train_pos_edge_index)
        pred = torch.sigmoid(pred)
        val_pred_pos = pred[self.data.val_pos_edge_index[0], self.data.val_pos_edge_index[1]]
        val_pred_neg = pred[self.data.val_neg_edge_index[0], self.data.val_neg_edge_index[1]]
        val_pred = torch.cat([val_pred_pos, val_pred_neg], dim=0).squeeze(-1).cpu()
        val_labels = torch.cat([torch.ones(val_pred_pos.shape), torch.zeros(val_pred_neg.shape)]).squeeze(-1).cpu()
        val_roc_auc = roc_auc_score(val_labels, val_pred)
        val_ap = average_precision_score(val_labels, val_pred)

        test_pred_pos = pred[self.data.test_pos_edge_index[0], self.data.test_pos_edge_index[1]]
        test_pred_neg = pred[self.data.test_neg_edge_index[0], self.data.test_neg_edge_index[1]]
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

    def info_nce_loss(self, features, pos_src, pos_tgt, neg_src, neg_tgt, temp=1.0):

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T).fill_diagonal_(0)

        positives = similarity_matrix[pos_src, pos_tgt]

        negatives = similarity_matrix[neg_src, neg_tgt]

        logits = torch.cat([positives, negatives], dim=0)
        logits = logits / temp

        labels = torch.cat([torch.ones_like(positives), torch.zeros_like(negatives)], dim=0)
        return logits, labels

class VGAENodeWrapper(TransNodeWrapper):
    def __init__(self, *args, **kwargs):
        super(VGAENodeWrapper, self).__init__(*args, **kwargs)
        self.new_index = None

    def __call__(self, x, edge_index):
        # new_index = self.permute_index(edge_index)
        return super(VGAENodeWrapper, self).__call__(x, edge_index)

    @torch.no_grad()
    def permute_index(self, previous_index):
        mu = self.embeddings.mu
        mu = torch.dropout(mu, 0.6, True)
        logvar = self.embeddings.logvar
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        new_adj = torch.sigmoid(torch.mm(z, z.t())).gt(0.99).float()
        new_index = torch_geometric.utils.dense_to_sparse(new_adj)[0]
        size = min(new_index.size(1), previous_index.size(1))
        sampled_index = random.sample([i for i in range(new_index.size(1))], size)
        sampled_new_index = new_index[:, sampled_index]

        index = torch.cat([previous_index, sampled_new_index], dim=-1)
        index = torch_geometric.utils.coalesce(index)
        return index


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


def create_vgae_task(model, batch_size, data, temp):
    return VGAELearner(model, batch_size, data, temp)

# def create_vge_node_transfer_task(model, predictor, embeddings, task, prompt_mode, concat_mode, k_prompts, data, dataset,
#                                   batch_size, type_trick, split_idx, prompt_type, prompt_raw, prompt_continual, **kwargs):
#     wrap = VGAENodeWrapper(model, predictor, embeddings, task, prompt_mode, concat_mode, data, k_prompts, prompt_type, prompt_raw,
#                            prompt_continual, **kwargs)
#     return NodeLearner(wrap, batch_size, data, dataset, type_trick, split_idx)
#
def create_vge_node_transfer_task(model, predictor, embeddings, args, task, data, split_idx):

    wrap = VGAENodeWrapper(model, predictor, embeddings, args, task, data)
    return NodeLearner(wrap, args.batch_size, data, args.dataset, args.type_trick, split_idx)
