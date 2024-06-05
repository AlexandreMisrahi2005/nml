import torch
from torch_geometric.nn import GraphConv, GCNConv, DimeNet
from torch_geometric.nn.pool import global_mean_pool

import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(Baseline, self).__init__()
        self.conv1 = GraphConv(num_features, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, 2 * hidden_dim)
        self.conv3 = GraphConv(2 * hidden_dim, 3 * hidden_dim)
        self.conv4 = GraphConv(3 * hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.head(x)
        return x
    
class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.head(x)
        return x
    
class DimeNetModel(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(DimeNetModel, self).__init__()
        self.dime = DimeNet(hidden_dim, out_channels=hidden_dim, num_blocks=3, num_bilinear=1, num_spherical=2, num_radial=2)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        z, pos, batch = data.z, data.pos, data.batch
        x = self.dime(z, pos, batch)
        x = global_mean_pool(x, data.batch)
        x = self.head(x)
        return x