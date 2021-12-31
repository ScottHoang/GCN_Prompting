from .GCN import GCN
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class VGAE(nn.Module):
    def __init__(self, args):
        super(VGAE, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.cached = False
        self.dim_hidden2 = self.dim_hidden//2
        self.encoder = GCNConv(self.num_feats, self.dim_hidden, cached=self.cached)
        self.gc1 = GCNConv(self.dim_hidden, self.dim_hidden2, cached=self.cached)
        self.gc2 = GCNConv(self.dim_hidden, self.dim_hidden2, cached=self.cached)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x, edge_index):
        z, mu, logvar = self.get_embeddings(x, edge_index)
        z = F.dropout(z, p=self.dropout, training=self.training)
        Adj = torch.mm(z, z.t())
        return Adj, mu, logvar

    def get_embeddings(self, x , edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.encoder(x, edge_index))

        mu = self.gc1(x, edge_index)
        logvar = self.gc2(x, edge_index)

        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
        else:
            z = mu

        return z, mu, logvar


