"""
Helper functions for scale-agnostic unconditional generation of material structures.
Implements utility functions for data processing, visualization, and model evaluation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

def setup_logging(log_dir: Path) -> None:
    """Configure logging with appropriate format and handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

def prepare_batch_graph(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Prepare a batch of graph data for model input.

    Args:
        node_features: Node feature tensor [num_nodes, num_features]
        edge_index: Edge connectivity tensor [2, num_edges]
        edge_attr: Optional edge attributes [num_edges, num_edge_features]
        device: Target device for tensors

    Returns:
        Dictionary containing prepared graph tensors
    """
    batch = {
        'node_features': node_features.to(device),
        'edge_index': edge_index.to(device),
    }
    if edge_attr is not None:
        batch['edge_attr'] = edge_attr.to(device)
    return batch

def compute_node_distances(
    positions: torch.Tensor,
    edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Compute pairwise distances between connected nodes.

    Args:
        positions: Node position tensor [num_nodes, 3]
        edge_index: Edge connectivity tensor [2, num_edges]

    Returns:
        Tensor of edge distances [num_edges]
    """
    row, col = edge_index
    dist = torch.norm(positions[row] - positions[col], dim=-1)
    return dist

def add_gaussian_noise(
    x: torch.Tensor,
    noise_std: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add Gaussian noise to input tensor.

    Args:
        x: Input tensor
        noise_std: Standard deviation of noise

    Returns:
        Tuple of (noisy tensor, noise tensor)
    """
    noise = torch.randn_like(x) * noise_std
    return x + noise, noise

def rbf_expansion(
    distances: torch.Tensor,
    num_rbf: int = 50,
    rbf_sigma: float = 0.1
) -> torch.Tensor:
    """
    Expand distances using radial basis functions.

    Args:
        distances: Edge distance tensor [num_edges]
        num_rbf: Number of RBF kernels
        rbf_sigma: Width of RBF kernels

    Returns:
        RBF expanded features [num_edges, num_rbf]
    """
    rbf_centers = torch.linspace(0, 2, num_rbf).to(distances.device)
    rbf_centers = rbf_centers.view(1, -1)
    distances = distances.view(-1, 1)

    rbf_features = torch.exp(-(distances - rbf_centers)**2 / (2 * rbf_sigma**2))
    return rbf_features

def compute_structure_statistics(
    positions: torch.Tensor,
    edge_index: torch.Tensor
) -> Dict[str, float]:
    """
    Compute basic statistical measures of the generated structure.

    Args:
        positions: Node position tensor [num_nodes, 3]
        edge_index: Edge connectivity tensor [2, num_edges]

    Returns:
        Dictionary of computed statistics
    """
    distances = compute_node_distances(positions, edge_index)

    stats = {
        'mean_distance': distances.mean().item(),
        'std_distance': distances.std().item(),
        'min_distance': distances.min().item(),
        'max_distance': distances.max().item(),
        'num_nodes': positions.shape[0],
        'num_edges': edge_index.shape[1]
    }
    return stats

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path
) -> None:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoint
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)

    logging.info(f'Saved checkpoint to {checkpoint_path}')

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """
    Load model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint file

    Returns:
        Tuple of (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint['epoch'], checkpoint['loss']
