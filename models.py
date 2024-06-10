from torch_geometric.nn import GraphConv, DimeNet, GraphSAGE, GAT
from torch_geometric.nn.pool import global_mean_pool

import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    """
    Baseline model with 4 GraphConv layers
    """
    def __init__(self, num_features, hidden_dim):
        """
        Initialize the model
        Args:
            num_features: Number of input features
            hidden_dim: Number of hidden units
        """
        super(Baseline, self).__init__()
        self.conv1 = GraphConv(num_features, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, 2 * hidden_dim)
        self.conv3 = GraphConv(2 * hidden_dim, 3 * hidden_dim)
        self.conv4 = GraphConv(3 * hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Forward pass of the model
        Args:
            data: PyTorch Geometric data object with node features x [num_nodes, num_features] and edge index [2, num_edges]
        Returns:
            x: Output of the model
        """
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
    """
    DimeNet model with 3 DimeNet layers
    """
    def __init__(self, hidden_dim):
        """
        Initialize the model
        Args:
            hidden_dim: Number of hidden units
        """
        super(DimeNetModel, self).__init__()
        self.dime = DimeNet(hidden_dim, out_channels=hidden_dim, num_blocks=3, num_bilinear=1, num_spherical=2, num_radial=2)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Forward pass of the model
        Args:
            data: PyTorch Geometric data object with node features z [num_nodes, num_features], pos [num_nodes, 3] and batch [num_nodes]
        Returns:
            x: Output of the model"""
        z, pos, batch = data.z, data.pos, data.batch
        x = self.dime(z, pos, batch)
        x = self.head(x)
        return x
    
class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model with 4 GraphSAGE layers
    """
    def __init__(self, num_features, hidden_dim):
        """
        Initialize the model
        Args:
            num_features: Number of input features
            hidden_dim: Number of hidden units
        """
        super(GraphSAGEModel, self).__init__()
        self.conv1 = GraphSAGE(in_channels=num_features, hidden_channels=hidden_dim, num_layers=4, dropout=0.2)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Forward pass of the model
        Args:
            data: PyTorch Geometric data object with node features x [num_nodes, num_features] and edge index [2, num_edges]
        Returns:
            x: Output of the model
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.head(x)
        return x
    
class GATModel(nn.Module):
    """
    GAT model with 4 GAT layers
    """
    def __init__(self, num_features, hidden_dim):
        """
        Initialize the model
        Args:
            num_features: Number of input features
            hidden_dim: Number of hidden units
        """
        super(GATModel, self).__init__()
        self.conv1 = GAT(in_channels=num_features, hidden_channels=hidden_dim, num_layers=4, dropout=0.2)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Forward pass of the model
        Args:
            data: PyTorch Geometric data object with node features x [num_nodes, num_features] and edge index [2, num_edges]
        Returns:
            x: Output of the model
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.head(x)
        return x