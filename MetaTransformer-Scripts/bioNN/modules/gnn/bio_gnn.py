import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.data import Data

class BioMessagePassing(MessagePassing):
    """Biologically-inspired message passing layer with synaptic plasticity."""
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "add" aggregation for biological plausibility
        self.lin = nn.Linear(in_channels, out_channels)
        self.synaptic_plasticity = nn.Parameter(torch.ones(1))

    def forward(self, x, edge_index, edge_attr=None):
        # Synaptic weight modulation
        edge_weight = F.sigmoid(self.synaptic_plasticity) if edge_attr is None \
                     else edge_attr * F.sigmoid(self.synaptic_plasticity)

        # Propagate messages
        return self.propagate(edge_index, size=(x.size(0), x.size(0)),
                            x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # Apply synaptic weights to messages
        return x_j * edge_weight.view(-1, 1)

class BioScaleGNN(nn.Module):
    """Scale-agnostic biological neural network."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers

        # Input processing
        self.input_transform = nn.Linear(input_dim, hidden_dim)

        # Bio message passing layers
        self.conv_layers = nn.ModuleList([
            BioMessagePassing(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Attention mechanism for scale invariance
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)

        # Neuroplasticity mechanism
        self.plasticity = nn.Parameter(torch.ones(num_layers))

        # Output processing
        self.output_transform = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, edge_index):  # Change from (self, data)
        # Initial transformation
        x = self.input_transform(x)

        # Process through bio message passing layers
        for i in range(self.num_layers):
            # Apply plasticity
            x = x * F.sigmoid(self.plasticity[i])

            # Message passing
            x = self.conv_layers[i](x, edge_index)

            # Apply non-linearity (membrane potential simulation)
            x = F.leaky_relu(x)

        # Apply attention for scale invariance
        x = x.unsqueeze(0)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)

        # Output transformation
        return self.output_transform(x)
def create_bio_gnn(input_dim, hidden_dim=64, output_dim=32, num_layers=3):
    """Factory function to create a BioScaleGNN instance."""
    model = BioScaleGNN(input_dim, hidden_dim, output_dim, num_layers)
    return model
