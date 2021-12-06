import importlib

import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator

from Dataloader import load_data, load_ogbn, prepare_edge_data
from tricks import TricksComb #TricksCombSGC
from utils import AcontainsB
from utils import LinkPredictor
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


class trainer(object):
    def __init__(self, args, which_run):
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.args = args
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        self.which_run = which_run
        # set dataloader
        self.set_dataloader()
        #
        if args.task == 'edge':
            args.num_classes = args.dim_hidden
            self.edge_predictor = LinkPredictor(args.dim_hidden, args.dim_hidden, 1, 3,
                                                args.dropout).to(self.device)
            self.optimizer_edge = self.edge_predictor.optimizer

        if args.compare_model:  # only compare model
            Model = getattr(importlib.import_module("models"), self.type_model)
            self.model = Model(args)
        else:  # compare tricks combinations
            self.model = TricksComb(args)
        self.model.to(self.device)
        self.optimizer = self.model.optimizer

    def set_dataloader(self):
        if self.args.task == 'node':
            if self.dataset == 'ogbn-arxiv':
                self.data, self.split_idx = load_ogbn(self.dataset)
                self.data.to(self.device)
                self.train_idx = self.split_idx['train'].to(self.device)
                self.evaluator = Evaluator(name='ogbn-arxiv')
                self.loss_fn = torch.nn.functional.nll_loss
            else:
                self.data = load_data(self.dataset, self.which_run)
                self.loss_fn = torch.nn.functional.nll_loss
                self.data.to(self.device)
        else:
            self.data = prepare_edge_data(self.args, self.which_run).to(self.device)
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def train_and_test(self):
        best_stats = None
        patience = self.args.patience
        bad_counter = 0.
        # val_loss_history = []
        for epoch in range(self.args.epochs):
            stats = self.train_net()
            best_stats = stats if best_stats is None else best_stats
            if self.args.task == 'node':
                # acc_train, acc_val, acc_test, loss_train, loss_val = self.train_net()
                # val_loss_history.append(loss_val)
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
            elif self.args.task == 'edge':
                if stats['val_roc_auc'] > best_stats['val_roc_auc']:
                    best_stats.update(stats)
                    bad_counter = 0
                else:
                    bad_counter += 1
            else:
                raise NotImplementedError
                # if epoch % 20 == 0:
            if epoch % 20 == 0 or epoch == 1:
                log = f"Epoch: {epoch:03d}, "
                for k, v in best_stats.items():
                    log = log + f"{k}: {v:.4f}, "
                print(log)
            if bad_counter == patience:
               break

        log = ''
        for k, v in best_stats.items():
            log = log + f"{k}: {v:.4f}, "
        print(log)
        return best_stats

    def train_net(self):
        try:
            loss_train = self.run_trainSet()
            stats = self.run_testSet()
            stats['loss_train'] = loss_train
            return stats

            # return acc_train, acc_val, acc_test, loss_train, loss_val
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
            else:
                raise e

    def run_trainSet(self):
        if self.args.task == 'node':
            return self.node_train()
        elif self.args.task == 'edge':
            return self.edge_train()
        elif self.args.task == 'prompt':
            pass
        else:
            raise NotImplementedError

    def node_train(self):
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

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def edge_train(self):
        self.model.train()
        self.edge_predictor.train()
        loss_epoch = []

        for perm in DataLoader(range(self.data.train_pos.size(1)), self.args.batch_size,
                               shuffle=True):
            embs = self.model(self.data.x, self.data.edge_index)

            pos_prediction = self.edge_predictor(embs[self.data.train_pos[0,perm]],
                                                  embs[self.data.train_pos[1,perm]])
            neg_prediction = self.edge_predictor(embs[self.data.train_neg[0,perm]],
                                                  embs[self.data.train_neg[1,perm]])

            loss = self.loss_fn(pos_prediction, torch.ones(pos_prediction.shape).to(self.device)) + \
                   self.loss_fn(neg_prediction, torch.zeros(neg_prediction.shape).to(self.device))

            self.optimizer.zero_grad()
            self.optimizer_edge.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_edge.step()
            loss_epoch.append(loss.item())
        mean_loss = sum(loss_epoch) / len(loss_epoch)
        return mean_loss

    def edge_test(self):
        self.model.eval()
        self.edge_predictor.eval()
        val_pred_pos = []
        val_pred_neg = []
        for perm in DataLoader(range(self.data.val_pos.size(1)), self.args.batch_size,
                               shuffle=False):

            embs = self.model(self.data.x, self.data.edge_index)
            #  val
            val_pred_pos.append(torch.sigmoid(self.edge_predictor(embs[self.data.val_pos[0,perm]],
                                                             embs[self.data.val_pos[1,perm]])))
            val_pred_neg.append(torch.sigmoid(self.edge_predictor(embs[self.data.val_neg[0,perm]],
                                                             embs[self.data.val_neg[1,perm]])))
        val_pred_pos = torch.cat(val_pred_pos, dim=0)
        val_pred_neg = torch.cat(val_pred_neg, dim=0)
        val_pred = torch.cat([val_pred_pos, val_pred_neg], dim=0).squeeze(-1).cpu()
        val_labels = torch.cat([torch.ones(val_pred_pos.shape), torch.zeros(val_pred_neg.shape)]).squeeze(-1).cpu()
        val_roc_auc = roc_auc_score(val_labels, val_pred)
        val_ap = average_precision_score(val_labels, val_pred)
        #  test
        test_pred_pos = []
        test_pred_neg = []
        for perm in DataLoader(range(self.data.test_pos.size(1)), self.args.batch_size,
                               shuffle=False):
            test_pred_pos.append(torch.sigmoid(self.edge_predictor(embs[self.data.test_pos[0,perm]],
                                                              embs[self.data.test_pos[1, perm]])))
            test_pred_neg.append(torch.sigmoid(self.edge_predictor(embs[self.data.test_neg[0,perm]],
                                                              embs[self.data.test_neg[1,perm]])))

        test_pred_pos = torch.cat(test_pred_pos, dim=0)
        test_pred_neg = torch.cat(test_pred_neg, dim=0)
        test_pred = torch.cat([test_pred_pos, test_pred_neg], dim=0).squeeze(-1).cpu()
        test_labels = torch.cat([torch.ones(test_pred_pos.shape), torch.zeros(test_pred_neg.shape)]).squeeze(-1).cpu()
        test_roc_auc = roc_auc_score(test_labels, test_pred)
        test_ap = average_precision_score(test_labels, test_pred)

        return {'valid_acc': val_ap, 'val_roc_auc': val_roc_auc,
                'test_acc': test_ap, 'test_roc_auc': test_roc_auc}


    @torch.no_grad()
    def run_testSet(self):
        if self.args.task == 'node':
            return self.node_test()
        elif self.args.task == 'edge':
            return self.edge_test()
        elif self.args.task == 'prompt':
            pass
        else:
            raise NotImplementedError


    def node_test(self):
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

            return {'train_acc':train_acc, 'valid_acc':valid_acc, 'test_acc':test_acc, "val_los":0}

        else:
            logits = self.model(self.data.x, self.data.edge_index)
            logits = F.log_softmax(logits, 1)
            acc_train = evaluate(logits, self.data.y, self.data.train_mask)
            acc_val = evaluate(logits, self.data.y, self.data.val_mask)
            acc_test = evaluate(logits, self.data.y, self.data.test_mask)
            val_loss = self.loss_fn(logits[self.data.val_mask], self.data.y[self.data.val_mask])
            return {'train_acc':acc_train, 'valid_acc':acc_val,
                    'test_acc':acc_test, "val_los":val_loss.item()}
