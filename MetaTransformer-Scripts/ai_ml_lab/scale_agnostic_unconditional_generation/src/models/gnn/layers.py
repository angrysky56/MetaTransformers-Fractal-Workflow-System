"""
Scale-agnostic GNN layers implementation.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class ScaleAgnosticConv(MessagePassing):
    """
    Scale-agnostic graph convolution layer with attention mechanism.
    """
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Linear transformations for node features
        self.lin_q = nn.Linear(in_channels, heads * out_channels)
        self.lin_k = nn.Linear(in_channels, heads * out_channels)
        self.lin_v = nn.Linear(in_channels, heads * out_channels)
        
        # Output transformation
        self.lin_out = nn.Linear(heads * out_channels, out_channels)
        
        # Attention mechanism
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        nn.init.xavier_uniform_(self.att)
        
    def forward(self, x, edge_index, edge_attr=None, size=None):
        """
        Forward pass of the layer.
        
        Args:
            x (torch.Tensor): Node feature matrix
            edge_index (torch.Tensor): Graph connectivity
            edge_attr (torch.Tensor, optional): Edge features
            size (tuple, optional): Size of the target output
            
        Returns:
            torch.Tensor: Updated node features
        """
        # Transform node features
        q = self.lin_q(x).view(-1, self.heads, self.out_channels)
        k = self.lin_k(x).view(-1, self.heads, self.out_channels)
        v = self.lin_v(x).view(-1, self.heads, self.out_channels)
        
        # Propagate messages
        out = self.propagate(edge_index, q=q, k=k, v=v, size=size)
        
        # Apply output transformation
        out = self.lin_out(out.view(-1, self.heads * self.out_channels))
        
        return out
    
    def message(self, q_i, k_j, v_j, edge_index_i, size_i):
        """
        Compute messages between nodes.
        """
        # Compute attention scores
        alpha = (q_i * k_j).sum(dim=-1) / torch.sqrt(torch.tensor(self.out_channels))
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)
        
        # Apply attention to values
        return alpha * v_j
    
    def aggregate(self, inputs, index, dim_size=None):
        """
        Aggregate messages at target nodes.
        """
        # Sum aggregation with attention weights
        return super().aggregate(inputs, index, dim_size)

class LocalStructureLayer(nn.Module):
    """
    Layer for capturing local structural information.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        self.conv1 = ScaleAgnosticConv(in_channels, hidden_channels)
        self.conv2 = ScaleAgnosticConv(hidden_channels, hidden_channels)
        self.conv3 = ScaleAgnosticConv(hidden_channels, out_channels)
        
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(out_channels)
        
        self.act = nn.ReLU()
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass through local structure layer.
        """
        # First convolution block
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = self.act(x)
        
        # Second convolution block
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = self.act(x)
        
        # Third convolution block
        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)
        
        return x
