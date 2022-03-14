import collections
import os
import sys
import matplotlib.pyplot as plt
import torch
import torch_geometric.utils as tu
import os.path as osp
from utils import MAD
from utils import I2NR
import torch

res = 20
DATASETS = ["ACTOR", "TEXAS", "WISCONSIN",
            "CORNELL", 'AmazonComputers', 'Citeseer', 'Cora']
colors = ['lightcoral', 'orange', 'gold', 'cornflowerblue', 'violet', 'limegreen',
          'teal']

def update_edges_w_prompt(data):
    if data['prompts'] is not None:
        edge_index = data['edges']
        prompts_idx = data['prompts']
        prompt_k = prompts_idx.size(1)
        src = torch.arange(prompts_idx.size(0)).tile(dims=(prompt_k, 1)).t().reshape(1, -1).to(edge_index.device)
        tgt = prompts_idx.reshape(1, -1).to(edge_index.device)
        prompt_edge_index = torch.cat([src, tgt], dim=0).to(edge_index.device)
        edge_index = torch.cat([edge_index, prompt_edge_index], dim=-1)
        edge_index = tu.coalesce(edge_index)
        return edge_index, prompt_edge_index
    else:
        return data['edges'], None


if __name__ == "__main__":
    _, src, dst = sys.argv

    dirs = os.listdir(src) # list of different method used
    scan_datasets = set()
    # getting list of datasets
    for folder in dirs:
        dataset = os.listdir(osp.join(src, folder)) # dataset
        scan_datasets.update(dataset)
    #
    fig, axe = plt.subplots(1, 1)
    #
    local = 2
    rmt = 8
    #
    chosen_embs = 'node_embs'
    scan_datasets = list(scan_datasets)
    scan_datasets.sort()
    dirs.sort()
    bar_data =  collections.defaultdict(dict)
    for dataset in scan_datasets:
        shortest_distance = torch.load(osp.join('data', dataset, 'distance.pth'))
        local_mask = shortest_distance.le(local)
        rmt_mask = shortest_distance.gt(rmt)
        for model in dirs:
            modeldir = osp.join(src, model)
            gap = 0
            if dataset in os.listdir(modeldir):
                datadir = osp.join(modeldir, dataset)
                latest_run = os.listdir(datadir)[-1]
                #
                latest_embs = os.listdir(osp.join(datadir, latest_run))[-1]
                emb_data = torch.load(osp.join(datadir, latest_run, latest_embs))
                embs = emb_data[chosen_embs]
                nodes = torch.arange(embs.size(0))
                mad_local = MAD(embs, local_mask.to(embs.device))
                mad_rmt = MAD(embs, rmt_mask.to(embs.device))
                # global_mad = MAD(embs, torch.ones_like(shortest_distance).to(embs.device))
                gap = mad_rmt - mad_local
                bar_data[dataset][model] = gap
                # edge_index, prompt_edge_index = update_edges_w_prompt(emb_data)
                # if prompt_edge_index is not None:
                #     i2nr_edge, _ = I2NR(edge_index, emb_data['labels'], 1)
                #     i2nr_prompt, _ = I2NR(prompt_edge_index, emb_data['labels'], 1)
                #     print(model, dataset, i2nr_edge, i2nr_prompt)
    #
    # print(bar_data)
    keys = ['TEXAS', 'WISCONSIN', 'CORNELL', 'Citeseer', "Cora"]
    x_axis = torch.arange(len(keys))
    ours = [bar_data[k1][k2] for k1 in keys for k2 in bar_data[k1] if k2 != 'GCN.2.node']
    base = [bar_data[k1][k2] for k1 in keys for k2 in bar_data[k1] if k2 == 'GCN.2.node']

    plt.bar(x_axis - 0.2, base, 0.4, label = "GCN")
    plt.bar(x_axis + 0.2, ours, 0.4, label = "Ours")
    plt.xticks(x_axis, keys)
    plt.ylabel("MAD-gap")
    plt.xlabel("Dataset")
    plt.legend()
    # plt.show()
    plt.savefig("mad-gap.pdf")



