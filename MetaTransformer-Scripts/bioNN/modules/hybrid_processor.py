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
        
        # Initialize biological neural network
        self.bio_network = create_bio_gnn(
            input_dim=input_dim,
            hidden_dim=bio_hidden_dim,
            output_dim=bio_hidden_dim,
            num_layers=bio_layers
        )
        
        # Initialize quantum-biological bridge
        self.entropic_bridge = BioEntropicBridge(
            bio_dim=bio_hidden_dim,
            quantum_dim=quantum_dim
        )
        
        # Processing state
        self.current_quantum_state = None
        self.current_uncertainty = 0.0
        
    def process_step(self, x: torch.Tensor, 
                    edge_index: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Execute one step of hybrid processing."""
        
        # 1. Biological processing
        bio_state = self.bio_network(x, edge_index)
        
        # 2. Convert to quantum state
        quantum_state, _ = self.entropic_bridge(bio_state=bio_state)
        self.current_quantum_state = quantum_state
        
        # 3. Quantum to biological mapping with uncertainty
        output_state, uncertainty = self.entropic_bridge(quantum_state=quantum_state)
        self.current_uncertainty = uncertainty
        
        # Prepare metrics
        metrics = {
            'uncertainty': uncertainty,
            'bio_state_norm': torch.norm(bio_state).item(),
            'quantum_state_norm': torch.norm(quantum_state).item(),
            'output_state_norm': torch.norm(output_state).item()
        }
        
        return output_state, metrics
    
    def get_quantum_state(self) -> Tuple[torch.Tensor, float]:
        """Get current quantum state and uncertainty."""
        return self.current_quantum_state, self.current_uncertainty

def create_processor(input_dim: int, **kwargs) -> HybridBioQuantumProcessor:
    """Factory function to create a hybrid processor instance."""
    return HybridBioQuantumProcessor(input_dim, **kwargs)