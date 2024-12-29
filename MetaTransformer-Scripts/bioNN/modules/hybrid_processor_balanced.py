import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from .gnn.bio_gnn import BioScaleGNN, create_bio_gnn
from .entropy.balanced_entropic_bridge import BalancedBioEntropicBridge

class BalancedHybridProcessor:
    """Enhanced hybrid biological-quantum neural processor with balanced uncertainty."""
    
    def __init__(self,
                 input_dim: int,
                 bio_hidden_dim: int = 64,
                 quantum_dim: int = 32,
                 bio_layers: int = 3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.bio_network = create_bio_gnn(
            input_dim=input_dim,
            hidden_dim=bio_hidden_dim,
            output_dim=bio_hidden_dim,
            num_layers=bio_layers
        ).to(self.device)
        
        # Initialize balanced entropic bridge
        self.entropic_bridge = BalancedBioEntropicBridge(
            bio_dim=bio_hidden_dim,
            quantum_dim=quantum_dim
        ).to(self.device)
        
        # State tracking
        self.current_quantum_state = None
        self.current_uncertainty = 0.5  # Initialize at balanced point
        self.measurement_history = []
        
    def process_step(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process one step through the hybrid system with enhanced metrics."""
        # Move inputs to correct device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Biological processing through GNN
        bio_state = self.bio_network(x, edge_index)
        
        # Bridge to quantum domain
        quantum_state, uncertainty = self.entropic_bridge(bio_state=bio_state)
        self.current_quantum_state = quantum_state
        self.current_uncertainty = uncertainty
        
        # Track measurements
        self.measurement_history.append({
            'uncertainty': uncertainty,
            'bio_state_norm': torch.norm(bio_state).item(),
            'quantum_state_norm': torch.norm(quantum_state).item()
        })
        
        # Keep history manageable
        if len(self.measurement_history) > 10:
            self.measurement_history.pop(0)
        
        # Calculate additional metrics
        metrics = {
            'uncertainty': uncertainty,
            'bio_state_norm': torch.norm(bio_state).item(),
            'quantum_state_norm': torch.norm(quantum_state).item(),
            'avg_uncertainty': sum(h['uncertainty'] for h in self.measurement_history) / len(self.measurement_history),
            'state_stability': self._calculate_stability()
        }
        
        return quantum_state, metrics
    
    def _calculate_stability(self) -> float:
        """Calculate stability metric from measurement history."""
        if len(self.measurement_history) < 2:
            return 1.0
            
        # Calculate variation in uncertainties
        uncertainties = [h['uncertainty'] for h in self.measurement_history]
        return 1.0 - torch.std(torch.tensor(uncertainties)).item()
    
    def get_quantum_state(self) -> Tuple[torch.Tensor, float]:
        """Get current quantum state and uncertainty."""
        return self.current_quantum_state, self.current_uncertainty
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        if not self.measurement_history:
            return {}
            
        return {
            'current_uncertainty': self.current_uncertainty,
            'avg_uncertainty': sum(h['uncertainty'] for h in self.measurement_history) / len(self.measurement_history),
            'min_uncertainty': min(h['uncertainty'] for h in self.measurement_history),
            'max_uncertainty': max(h['uncertainty'] for h in self.measurement_history),
            'stability': self._calculate_stability(),
            'measurement_count': len(self.measurement_history)
        }

def create_balanced_processor(input_dim: int, **kwargs) -> BalancedHybridProcessor:
    """Factory function to create a balanced hybrid processor instance."""
    return BalancedHybridProcessor(input_dim, **kwargs)