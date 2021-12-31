import os
import random
from warnings import warn

import torch as th
import yaml
from texttable import Texttable
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch
import numpy as np
from torch_geometric.utils import to_undirected, from_scipy_sparse_matrix, is_undirected
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import sys
import os.path

from sklearn.model_selection import StratifiedShuffleSplit

cur_dir = os.path.dirname(os.path.realpath(__file__))
par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append('%s/software/' % par_dir)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_args(args):
    _dict = vars(args)
    _key = sorted(_dict.items(), key=lambda x: x[0])
    t = Texttable()
    t.add_row(["Parameter", "Value"])
    for k, _ in _key:
        t.add_row([k, _dict[k]])
    print(t.draw())


def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


def AcontainsB(A, listB):
    # A: string; listB: list of strings
    for s in listB:
        if s in A: return True
    return False


def yaml_parser(model):
    filename = os.path.join('options/configs', f'{model}.yml')
    if os.path.exists(filename):
        with open(filename, 'r') as yaml_f:
            configs = yaml.load(yaml_f, Loader=yaml.FullLoader)
        return configs
    else:
        warn(f'configs of {model} not found, use the default setting instead')
        return {}


def overwrite_with_yaml(args, model, dataset):
    configs = yaml_parser(model)
    if dataset not in configs.keys():
        warn(f'{model} have no specific settings on {dataset}. Use the default setting instead.')
        return args
    for k, v in configs[dataset].items():
        if k in args.__dict__:
            args.__dict__[k] = v
        else:
            warn(f"Ignored unknown parameter {k} in yaml.")
    return args


class TaskPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, lr, weight_decay):
        super(TaskPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        # self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        hidden_channels = in_channels
        for _ in range(num_layers-1):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
            hidden_channels = hidden_channels // 2
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j=None):
        if x_j is not None:
            x = x_i * x_j
        else:
            x = x_i
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class CompoundOptimizers():
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


def floor(x):
    return torch.div(x, 1, rounding_mode='trunc')


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def split_edges(data, args):
    # set_random_seed(args.seed)
    row, col = data.edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    n_v = floor(args.val_ratio * row.size(0)).int()  # number of validation positive edges
    n_t = floor(args.test_ratio * row.size(0)).int()  # number of test positive edges
    # split positive edges
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos = torch.stack([r, c], dim=0)

    # sample negative edges
    if args.practical_neg_sample == False:
        # If practical_neg_sample == False, the sampled negative edges
        # in the training and validation set aware the test set

        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample all the negative edges and split into val, test, train negs
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.randperm(neg_row.size(0))[:row.size(0)]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v + n_t:], neg_col[n_v + n_t:]
        data.train_neg = torch.stack([row, col], dim=0)

    else:
        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample the test negative edges first
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.randperm(neg_row.size(0))[:n_t]
        neg_row, neg_col = neg_row[perm], neg_col[perm]
        data.test_neg = torch.stack([neg_row, neg_col], dim=0)

        # Sample the train and val negative edges with only knowing
        # the train positive edges
        row, col = data.train_pos
        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample the train and validation negative edges
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()

        n_tot = n_v + data.train_pos.size(1)
        perm = torch.randperm(neg_row.size(0))[:n_tot]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:], neg_col[n_v:]
        data.train_neg = torch.stack([row, col], dim=0)

    return data


def k_hop_subgraph(node_idx, num_hops, edge_index, max_nodes_per_hop=None, num_nodes=None):
    if num_nodes == None:
        num_nodes = torch.max(edge_index) + 1
    row, col = edge_index
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    if max_nodes_per_hop == None:
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
    else:
        not_visited = row.new_empty(num_nodes, dtype=torch.bool)
        not_visited.fill_(True)
        for _ in range(num_hops):
            node_mask.fill_(False)  # the source node mask in this hop
            node_mask[subsets[-1]] = True  # mark the sources
            not_visited[subsets[-1]] = False  # mark visited nodes
            torch.index_select(node_mask, 0, row, out=edge_mask)  # indices of all neighbors
            neighbors = col[edge_mask].unique()  # remove repeats
            neighbor_mask = row.new_empty(num_nodes, dtype=torch.bool)  # mask of all neighbor nodes
            edge_mask_hop = row.new_empty(row.size(0), dtype=torch.bool)  # selected neighbor mask in this hop
            neighbor_mask.fill_(False)
            neighbor_mask[neighbors] = True
            neighbor_mask = torch.logical_and(neighbor_mask, not_visited)  # all neighbors that are not visited
            ind = torch.where(neighbor_mask == True)  # indicies of all the unvisited neighbors
            if ind[0].size(0) > max_nodes_per_hop:
                perm = torch.randperm(ind[0].size(0))
                ind = ind[0][perm]
                neighbor_mask[ind[max_nodes_per_hop:]] = False  # randomly select max_nodes_per_hop nodes
                torch.index_select(neighbor_mask, 0, col, out=edge_mask_hop)  # find the indicies of selected nodes
                edge_mask = torch.logical_and(edge_mask, edge_mask_hop)  # change edge_mask
            subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    node_idx = row.new_full((num_nodes,), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


def plus_edge(data_observed, label, p_edge, args):
    nodes, edge_index_m, mapping, _ = k_hop_subgraph(node_idx=p_edge, num_hops=args.num_hops, \
                                                     edge_index=data_observed.edge_index,
                                                     max_nodes_per_hop=args.max_nodes_per_hop,
                                                     num_nodes=data_observed.num_nodes)
    x_sub = data_observed.x[nodes, :]
    edge_index_p = edge_index_m
    edge_index_p = torch.cat((edge_index_p, mapping.view(-1, 1)), dim=1)
    edge_index_p = torch.cat((edge_index_p, mapping[[1, 0]].view(-1, 1)), dim=1)

    # edge_mask marks the edge under perturbation, i.e., the candidate edge for LP
    edge_mask = torch.ones(edge_index_p.size(1), dtype=torch.bool)
    edge_mask[-1] = False
    edge_mask[-2] = False

    # if args.drnl == True:
    #     num_nodes = torch.max(edge_index_p) + 1
    #     z = drnl_node_labeling(edge_index_m, mapping[0], mapping[1], num_nodes)
    #     data = Data(edge_index=edge_index_p, x=x_sub, z=z)
    # else:
    data = Data(edge_index=edge_index_p, x=x_sub, z=0)
    data.edge_mask = edge_mask

    # label = 1 if the candidate link (p_edge) is positive and label=0 otherwise
    data.label = float(label)

    return data


def minus_edge(data_observed, label, p_edge, args):
    nodes, edge_index_p, mapping, _ = k_hop_subgraph(node_idx=p_edge, num_hops=args.num_hops, \
                                                     edge_index=data_observed.edge_index,
                                                     max_nodes_per_hop=args.max_nodes_per_hop,
                                                     num_nodes=data_observed.num_nodes)
    x_sub = data_observed.x[nodes, :]

    # edge_mask marks the edge under perturbation, i.e., the candidate edge for LP
    edge_mask = torch.ones(edge_index_p.size(1), dtype=torch.bool)
    ind = torch.where((edge_index_p == mapping.view(-1, 1)).all(dim=0))
    edge_mask[ind[0]] = False
    ind = torch.where((edge_index_p == mapping[[1, 0]].view(-1, 1)).all(dim=0))
    edge_mask[ind[0]] = False
    # if args.drnl == True:
    #     num_nodes = torch.max(edge_index_p) + 1
    #     z = drnl_node_labeling(edge_index_p[:, edge_mask], mapping[0], mapping[1], num_nodes)
    #     data = Data(edge_index=edge_index_p, x=x_sub, z=z)
    # else:
    data = Data(edge_index=edge_index_p, x=x_sub, z=0)
    data.edge_mask = edge_mask

    # label = 1 if the candidate link (p_edge) is positive and label=0 otherwise
    data.label = float(label)
    return data


def load_splitted_data(args):
    par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_name = args.dataset + '_split_' + args.data_split_num
    if args.test_ratio == 0.5:
        data_dir = os.path.join(par_dir, 'data/splitted_0_5/{}.mat'.format(data_name))
    else:
        data_dir = os.path.join(par_dir, 'data/splitted/{}.mat'.format(data_name))
    import scipy.io as sio
    print('Load data from: ' + data_dir)
    net = sio.loadmat(data_dir)
    data = Data()

    data.train_pos = torch.from_numpy(np.int64(net['train_pos']))
    data.train_neg = torch.from_numpy(np.int64(net['train_neg']))
    data.test_pos = torch.from_numpy(np.int64(net['test_pos']))
    data.test_neg = torch.from_numpy(np.int64(net['test_neg']))

    n_pos = floor(args.val_ratio * len(data.train_pos)).int()
    nlist = np.arange(len(data.train_pos))
    np.random.shuffle(nlist)
    val_pos_list = nlist[:n_pos]
    train_pos_list = nlist[n_pos:]
    data.val_pos = data.train_pos[val_pos_list]
    data.train_pos = data.train_pos[train_pos_list]

    n_neg = floor(args.val_ratio * len(data.train_neg)).int()
    nlist = np.arange(len(data.train_neg))
    np.random.shuffle(nlist)
    val_neg_list = nlist[:n_neg]
    train_neg_list = nlist[n_neg:]
    data.val_neg = data.train_neg[val_neg_list]
    data.train_neg = data.train_neg[train_neg_list]

    data.val_pos = torch.transpose(data.val_pos, 0, 1)
    data.val_neg = torch.transpose(data.val_neg, 0, 1)
    data.train_pos = torch.transpose(data.train_pos, 0, 1)
    data.train_neg = torch.transpose(data.train_neg, 0, 1)
    data.test_pos = torch.transpose(data.test_pos, 0, 1)
    data.test_neg = torch.transpose(data.test_neg, 0, 1)
    num_nodes = max(torch.max(data.train_pos), torch.max(data.test_pos)) + 1
    num_nodes = max(num_nodes, torch.max(data.val_pos) + 1)
    data.num_nodes = num_nodes

    return data


def load_unsplitted_data(args):
    # read .mat format files
    data_dir = os.path.join(par_dir, 'data/{}.mat'.format(args.dataset))
    print('Load data from: ' + data_dir)
    import scipy.io as sio
    net = sio.loadmat(data_dir)
    edge_index, _ = from_scipy_sparse_matrix(net['net'])
    data = Data(edge_index=edge_index)
    if is_undirected(data.edge_index) == False:  # in case the dataset is directed
        data.edge_index = to_undirected(data.edge_index)
    data.num_nodes = torch.max(data.edge_index) + 1
    return data


def load_Planetoid_data(args):
    print('Using data: ' + args.dataset)
    # dataset = Planetoid(root=par_dir+'/data/', name=args.dataset, transform=NormalizeFeatures())
    dataset = Planetoid(root=par_dir + '/data/', name=args.dataset)
    data = dataset[0]
    data.num_nodes = torch.max(data.edge_index) + 1
    return data


# def load_Planetoid_data(args):
#     print('downloading data: '+ args.dataset)
#     #dataset = Planetoid(root=par_dir+'/data/', name=args.dataset, transform=NormalizeFeatures())
#     dataset = Planetoid(root=par_dir+'/data/', name=args.dataset)
#     # Edited from https://github.com/tkipf/gae/blob/master/gae/input_data.py
#     names = ['x', 'tx', 'allx', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("./data/{}/raw/ind.{}.{}".format(args.dataset,args.data_name, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))
#     x, tx, allx, graph = tuple(objects)
#     filename="./data/{}/raw/ind.{}.test.index".format(args.dataset,args.data_name)
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     test_idx_reorder = index
#     test_idx_range = np.sort(test_idx_reorder)
#     if args.dataset == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     features=torch.tensor(sp.coo_matrix.todense(features)).float()
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#     edge_index=from_scipy_sparse_matrix(adj)[0]
#     data=Data(edge_index=edge_index,x=features)
#     data.num_nodes = torch.max(data.edge_index)+1

#     return data


def set_init_attribute_representation(data, args):
    # Construct data_observed and compute its node attributes & representation
    edge_index = torch.cat((data.train_pos, data.train_pos[[1, 0], :]), dim=1)
    if args.observe_val_and_injection == False:
        data_observed = Data(edge_index=edge_index)
    else:
        edge_index = torch.cat((edge_index, data.val_pos, data.val_pos[[1, 0], :]), dim=1)
        data_observed = Data(edge_index=edge_index)
    data_observed.num_nodes = data.num_nodes
    if args.observe_val_and_injection == False:
        edge_index_observed = data_observed.edge_index
    else:
        # use the injection trick and add val data in observed graph
        edge_index_observed = torch.cat((data_observed.edge_index, \
                                         data.train_neg, data.train_neg[[1, 0], :], data.val_neg,
                                         data.val_neg[[1, 0], :]), dim=1)
    # generate the initial node attribute if there isn't any
    if data.x == None:
        if args.init_attribute == 'n2v':
            from node2vec import CalN2V
            x = CalN2V(edge_index_observed, args)
        if args.init_attribute == 'one_hot':
            x = F.one_hot(torch.arange(data.num_nodes), num_classes=data.num_nodes)
            x = x.float()
        if args.init_attribute == 'spc':
            from SPC import spc
            x = spc(edge_index_observed, args)
            x = x.float()
        if args.init_attribute == 'ones':
            x = torch.ones(data.num_nodes, args.embedding_dim)
            x = x.float()
        if args.init_attribute == 'zeros':
            x = torch.zeros(data.num_nodes, args.embedding_dim)
            x = x.float()
    else:
        x = data.x
    # generate the initial node representation using unsupervised models
    # if args.init_representation != None:
    #     val_and_test = [data.test_pos, data.test_neg, data.val_pos, data.val_neg]
    #     num_nodes, _ = x.shape
    #     # add self-loop for the last node to aviod losing node if the last node dosen't have a link.
    #     if (num_nodes - 1) in edge_index_observed:
    #         edge_index_observed = edge_index_observed.clone().detach()
    #     else:
    #         edge_index_observed = torch.cat(
    #             (edge_index_observed.clone().detach(), torch.tensor([[num_nodes - 1], [num_nodes - 1]])), dim=1)
    #     if args.init_representation == 'gic':
    #         args.par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    #         sys.path.append('%s/software/GIC/' % args.par_dir)
    #         from GICEmbs import CalGIC
    #         data_observed.x, auc, ap = CalGIC(edge_index_observed, x, args.dataset, val_and_test, args)
    #
    #     if args.init_representation == 'vgae':
    #         from vgae import CalVGAE
    #         data_observed.x, auc, ap = CalVGAE(edge_index_observed, x, val_and_test, args)
    #     if args.init_representation == 'svgae':
    #         from svgae import CalSVGAE
    #         data_observed.x, auc, ap = CalSVGAE(edge_index_observed, x, val_and_test, args)
    #     if args.init_representation == 'argva':
    #         from argva import CalARGVA
    #         data_observed.x, auc, ap = CalARGVA(edge_index_observed, x, val_and_test, args)
    #     feature_results = [auc, ap]
    # else:
    data_observed.x = x
    feature_results = None

    return data_observed, feature_results


class StratifiedSampler():
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = th.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`
    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor
    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y

    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def corrcoef(x):
    """
    Mimics `np.corrcoef`
    Arguments
    ---------
    x : 2D torch.Tensor

    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)
    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref:
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013
    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c

@torch.no_grad()
def MAD(embeddings, targets=None):
    n = embeddings.size(0)#
    dij = []
    for i in range(n):
        dij.append(1 - torch.cosine_similarity(embeddings[i].unsqueeze(0), embeddings, dim=-1).unsqueeze(0))
    dij = torch.cat(dij, dim=0)
    dtgt = torch.mul(dij, targets)
    dtgt = dtgt.sum(dim=-1) / targets.sum(dim=-1)
    return dtgt.mean().item()

@torch.no_grad()
def I2NR(edges, labels, hops=2):
    edges = to_undirected(edges)
    i2nr = []
    nodes = set(edges.reshape(-1).cpu().tolist())
    for node in nodes:
        node_label = labels[node]
        subset, k_edges, _, _, = k_hop_subgraph(node, hops, edges)
        source_labels = labels[k_edges[0, :]]
        target_labels = labels[k_edges[1, :]]
        like_edges = source_labels.eq(target_labels)
        tgt = source_labels.eq(node_label)
        information = torch.mul(like_edges, tgt).sum()
        i2nr.append(information.div(k_edges.size(1)).item())
    mean_i2nr = sum(i2nr)/len(i2nr)
    missing_node_ratio = len(nodes) / labels.size(0)
    weighted_mean = mean_i2nr * missing_node_ratio
    return mean_i2nr, weighted_mean
