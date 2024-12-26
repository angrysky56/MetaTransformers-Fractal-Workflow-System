"""
Feature encoders for scale-agnostic GNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeEncoder(nn.Module):
    """
    Encoder for node features with support for categorical and continuous features.
    Implements scale-invariant feature processing.
    """
    def __init__(self, num_categorical_features, num_continuous_features, 
                 embedding_dim, dropout=0.1):
        super().__init__()
        
        self.num_categorical_features = num_categorical_features
        self.num_continuous_features = num_continuous_features
        self.embedding_dim = embedding_dim
        
        # Embeddings for categorical features
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=256, embedding_dim=embedding_dim)
            for _ in range(num_categorical_features)
        ])
        
        # Scale-aware continuous feature processing
        self.continuous_encoder = nn.Sequential(
            nn.Linear(num_continuous_features, embedding_dim * num_continuous_features),
            nn.LayerNorm(embedding_dim * num_continuous_features),
            nn.ReLU()
        )
        
        # Output transformation with scale normalization
        total_features = embedding_dim * (num_categorical_features + num_continuous_features)
        self.output_transform = nn.Sequential(
            nn.Linear(total_features, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, categorical_features, continuous_features):
        """
        Forward pass of the encoder.
        
        Args:
            categorical_features: Tensor [num_nodes, num_categorical_features]
            continuous_features: Tensor [num_nodes, num_continuous_features]
            
        Returns:
            Tensor [num_nodes, embedding_dim]
        """
        batch_size = continuous_features.size(0)
        
        # Encode categorical features
        categorical_embeddings = []
        for i, embedding_layer in enumerate(self.categorical_embeddings):
            feature = categorical_features[:, i]
            embedded = embedding_layer(feature)
            categorical_embeddings.append(embedded)
        
        # Process categorical features
        if categorical_embeddings:
            categorical_encoded = torch.cat(categorical_embeddings, dim=-1)
        else:
            categorical_encoded = torch.empty(
                (batch_size, 0), device=continuous_features.device
            )
        
        # Process continuous features with scale awareness
        continuous_encoded = self.continuous_encoder(continuous_features)
        
        # Combine features
        combined_features = torch.cat([categorical_encoded, continuous_encoded], dim=-1)
        
        # Final transformation
        encoded_features = self.output_transform(combined_features)
        
        return encoded_features

class EdgeEncoder(nn.Module):
    """
    Scale-aware encoder for edge features supporting both geometric and topological properties.
    """
    def __init__(self, in_channels, hidden_dim, out_channels, num_distance_bins=16):
        super().__init__()
        
        self.num_distance_bins = num_distance_bins
        
        # Distance encoding
        self.distance_embedding = nn.Embedding(num_distance_bins, hidden_dim)
        
        # Feature processing layers
        self.feature_encoder = nn.Sequential(
            nn.Linear(in_channels + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Scale-aware transformation
        self.scale_aware_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )
        
    def forward(self, edge_features, edge_distances):
        """
        Forward pass through edge encoder.
        
        Args:
            edge_features: Tensor [num_edges, in_channels]
            edge_distances: Tensor [num_edges]
            
        Returns:
            Tensor [num_edges, out_channels]
        """
        # Discretize distances into bins
        distance_bins = torch.bucketize(
            edge_distances, 
            torch.linspace(0, edge_distances.max(), self.num_distance_bins + 1)
        )
        
        # Get distance embeddings
        distance_embedded = self.distance_embedding(distance_bins)
        
        # Combine with edge features
        combined_features = torch.cat([edge_features, distance_embedded], dim=-1)
        
        # Process features
        encoded_features = self.feature_encoder(combined_features)
        
        # Apply scale-aware transformation
        output = self.scale_aware_transform(encoded_features)
        
        return output

class StructureEncoder(nn.Module):
    """
    Complete structure encoder combining node and edge features with scale awareness.
    """
    def __init__(self, node_encoder, edge_encoder):
        super().__init__()
        
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        
    def forward(self, node_categorical, node_continuous, edge_features, edge_distances):
        """
        Encode complete structure with scale awareness.
        
        Args:
            node_categorical: Categorical node features
            node_continuous: Continuous node features
            edge_features: Edge features
            edge_distances: Edge distances
            
        Returns:
            tuple: (encoded_nodes, encoded_edges)
        """
        # Encode nodes
        encoded_nodes = self.node_encoder(node_categorical, node_continuous)
        
        # Encode edges
        encoded_edges = self.edge_encoder(edge_features, edge_distances)
        
        return encoded_nodes, encoded_edges
