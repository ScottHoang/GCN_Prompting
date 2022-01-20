import os

import networkx as nx
import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
import torch_geometric.utils as U
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Coauthor, WebKB, Actor, Amazon
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, to_networkx
from utils import *

from sklearn.decomposition import  PCA

def load_ogbn(dataset='ogbn-arxiv'):
    dataset = PygNodePropPredDataset(name=dataset)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    return data, split_idx


def random_coauthor_amazon_splits(data):
    # https://github.com/mengliu1998/DeeperGNN/blob/da1f21c40ec535d8b7a6c8127e461a1cd9eadac1/DeeperGNN/train_eval.py#L17
    num_classes, lcc = data.num_classes, data.lcc
    lcc_mask = None
    if lcc:  # select largest connected component
        data_nx = to_networkx(data)
        data_nx = data_nx.to_undirected()
        print("Original #nodes:", data_nx.number_of_nodes())
        data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
        print("#Nodes after lcc:", data_nx.number_of_nodes())
        lcc_mask = list(data_nx.nodes)

    def index_to_mask(index, size):
        mask = torch.zeros(size, dtype=torch.bool, device=index.device)
        mask[index] = 1
        return mask

    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data


def manual_split_WebKB_Actor(data, which_split):
    # which_split take values from 0 to 9, type is int
    assert which_split in np.arange(10, dtype=int).tolist()

    data.train_mask = data.train_mask[:, which_split]
    data.val_mask = data.val_mask[:, which_split]
    data.test_mask = data.test_mask[:, which_split]
    return data


def change_split(data, dataset, which_split=0):
    if dataset in ["CoauthorCS", "CoauthorPhysics"]:
        data = random_coauthor_amazon_splits(data)
    elif dataset in ["AmazonComputers", "AmazonPhoto"]:
        data = random_coauthor_amazon_splits(data)
    elif dataset in ["TEXAS", "WISCONSIN", "CORNELL"]:
        data = manual_split_WebKB_Actor(data, which_split)
    elif dataset == "ACTOR":
        data = manual_split_WebKB_Actor(data, which_split)
    else:
        data = data
    data.y = data.y.long()
    return data


def load_data(dataset, which_run, norm=T.NormalizeFeatures()):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', dataset)

    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        data = Planetoid(path, dataset, split='public', transform=norm)[0]
        data.num_nodes = torch.max(data.edge_index) + 1

    elif dataset in ["CoauthorCS", "CoauthorPhysics"]:
        data = Coauthor(path, dataset[8:], transform=norm)[0]
        data.num_classes = int(max(data.y) + 1)
        data.lcc = False
        data = change_split(data, dataset, which_split=int(which_run // 10))

    elif dataset in ["AmazonComputers", "AmazonPhoto"]:
        data = Amazon(path, dataset[6:], transform=norm)[0]
        data.num_classes = int(max(data.y) + 1)
        data.lcc = True
        data = change_split(data, dataset, which_split=int(which_run // 10))

    elif dataset in ["TEXAS", "WISCONSIN", "CORNELL"]:
        data = WebKB(path, dataset, transform=norm)[0]
        data = change_split(data, dataset, which_split=int(which_run // 10))

    elif dataset == "ACTOR":
        data = Actor(path, transform=norm)[0]
        data = change_split(data, dataset, which_split=int(which_run // 10))

    else:
        raise Exception(f'the dataset of {dataset} has not been implemented')

    num_nodes = data.x.size(0)
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
    if isinstance(edge_index, tuple):
        data.edge_index = edge_index[0]
    else:
        data.edge_index = edge_index
    return data

def prepare_data(data, args, which_run):
    data.pca = torch.tensor(PCA(n_components=args.prompt_pca).fit_transform(data.x.detach().cpu().numpy())).to(data.x.device)
    edge_index = data.edge_index.clone()
    # train_test_split = T.RandomLinkSplit(is_undirected=True, num_val=args.val_ratio, num_test=args.test_ratio, split_labels=True)
    # train_data, val_data, test_data = train_test_split(data)
    # import pdb; pdb.set_trace()

    data = split_edges(data, args)
    data.edge_index = edge_index
    print_data_stats(data)
    print(f"data stats: TotalEdges {data.edge_index.size(1)}, trainEdges: {data.train_pos_edge_index.size(1)}, "
          f"ValEdges: {data.val_pos_edge_index.size(1) + data.val_neg_edge_index.size(1)}, "
          f"TestEdges: {data.test_pos_edge_index.size(1) + data.test_neg_edge_index.size(1)}")
    return data

def print_data_stats(data):
    num_nodes = data.x.size(0)
    num_train = data.train_mask.sum(0)
    num_val = data.val_mask.sum(0)
    num_test = data.test_mask.sum(0)

    homophily = torch_geometric.utils.homophily(data.edge_index , data.y)

    msg = f"nodes: {num_nodes}, trainNode: {num_train}, valNode: {num_val}, testNode: {num_test}, homophily: {homophily}"
    print(msg)
