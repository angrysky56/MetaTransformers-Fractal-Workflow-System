import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from .gnn.bio_gnn import BioScaleGNN, create_bio_gnn
from .entropy.entropic_bridge import BioEntropicBridge

class HybridBioQuantumProcessor:
    """Manages the hybrid biological-quantum neural processing pipeline."""
    def __init__(self,
                 input_dim: int,
                 bio_hidden_dim: int = 64,
                 quantum_dim: int = 32,
                 bio_layers: int = 3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize networks on correct device
        self.bio_network = create_bio_gnn(
            input_dim=input_dim,
            hidden_dim=bio_hidden_dim,
            output_dim=bio_hidden_dim,
            num_layers=bio_layers
        ).to(self.device)

        self.entropic_bridge = BioEntropicBridge(
            bio_dim=bio_hidden_dim,
            quantum_dim=quantum_dim
        ).to(self.device)

        self.current_quantum_state = None
        self.current_uncertainty = 0.0

    def process_step(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process one step through the hybrid system"""
        # Move inputs to correct device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        # Biological processing
        bio_state = self.bio_network(x, edge_index)

        # Quantum bridge processing
        quantum_state, uncertainty = self.entropic_bridge(bio_state)
        self.current_quantum_state = quantum_state
        self.current_uncertainty = uncertainty

        # Prepare metrics - handle both tensor and float cases
        metrics = {
            'uncertainty': uncertainty if isinstance(uncertainty, float) else uncertainty.item(),
            'bio_state_norm': torch.norm(bio_state).item(),
            'quantum_state_norm': torch.norm(quantum_state).item()
        }

        return quantum_state, metrics

    def get_quantum_state(self) -> Tuple[torch.Tensor, float]:
     """Get current quantum state and uncertainty."""
    def get_quantum_state(self) -> Tuple[torch.Tensor, float]:
        """Get current quantum state and uncertainty."""
        return self.current_quantum_state, self.current_uncertainty
def create_processor(input_dim: int, **kwargs) -> HybridBioQuantumProcessor:
    """Factory function to create a hybrid processor instance."""
    return HybridBioQuantumProcessor(input_dim, **kwargs)
