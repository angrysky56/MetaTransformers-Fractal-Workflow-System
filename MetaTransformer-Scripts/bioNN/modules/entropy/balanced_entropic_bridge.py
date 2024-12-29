"""
Balanced Entropic Bridge
Handles bio-quantum state transitions with entropy optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class EntropyConfig:
    """Configuration for entropy management."""
    dimension_depth: int = 3
    coherence_threshold: float = 0.92
    min_eigenvalue: float = 1e-7
    buffer_size: int = 5

class BalancedEntropicBridge(nn.Module):
    """Balanced bridge between biological and quantum states."""
    
    def __init__(self, dimension_depth: int = 3):
        super().__init__()
        self.config = EntropyConfig(dimension_depth=dimension_depth)
        self.measurement_buffer = []
    
    def calculate_density_matrix(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate density matrix with proper normalization."""
        # Ensure state is properly shaped
        if len(state.shape) > 1:
            state = state.reshape(-1)
        
        # Create density matrix
        density = torch.outer(state, state.conj())
        
        # Ensure Hermitian
        density = (density + density.conj().T) / 2
        
        # Normalize trace to 1
        trace = torch.trace(density).real
        if trace > 0:
            density = density / trace
        
        return density
    
    def measure_uncertainty(self, density_matrix: torch.Tensor) -> float:
        """Calculate uncertainty using eigenvalue distribution."""
        try:
            # Get eigenvalues
            eigenvalues = torch.linalg.eigvals(density_matrix).real
            
            # Filter and sort valid eigenvalues
            valid_eigs = eigenvalues[eigenvalues > self.config.min_eigenvalue]
            valid_eigs = torch.sort(valid_eigs, descending=True)[0]
            
            if len(valid_eigs) == 0:
                return 0.5  # Return mid-point if no valid eigenvalues
            
            # Normalize eigenvalues
            norm_eigs = valid_eigs / valid_eigs.sum()
            
            # Calculate von Neumann entropy
            entropy = -torch.sum(norm_eigs * torch.log2(norm_eigs + self.config.min_eigenvalue))
            
            # Scale entropy to [0, 1]
            max_entropy = np.log2(len(norm_eigs))
            if max_entropy > 0:
                scaled_uncertainty = float(entropy / max_entropy)
                
                # Apply sigmoid-like scaling to avoid extremes
                balanced_uncertainty = 0.5 * (1 + np.tanh(2 * scaled_uncertainty - 1))
                
                # Update buffer for smoothing
                self.measurement_buffer.append(balanced_uncertainty)
                if len(self.measurement_buffer) > self.config.buffer_size:
                    self.measurement_buffer.pop(0)
                
                # Return smoothed uncertainty
                return sum(self.measurement_buffer) / len(self.measurement_buffer)
            
        except Exception as e:
            print(f"Error in uncertainty calculation: {e}")
        
        return 0.5
    
    def measure_state(self, quantum_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Measure quantum state with balanced uncertainty."""
        density_matrix = self.calculate_density_matrix(quantum_state)
        uncertainty = self.measure_uncertainty(density_matrix)
        
        # Calculate observables
        measurement = torch.abs(quantum_state) ** 2
        
        return measurement, uncertainty

class BalancedBioEntropicBridge(nn.Module):
    """
    Enhanced bridge between biological and quantum processing.
    Uses balanced entropy measurements for state transitions.
    """
    
    def __init__(self, bio_dim: int, quantum_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.bio_dim = bio_dim
        self.quantum_dim = quantum_dim
        
        # Initialize entropy measurement
        self.measurement = BalancedEntropicBridge(quantum_dim)
        
        # Bio to quantum transformation
        self.bio_to_quantum = nn.Sequential(
            nn.Linear(bio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),  # Using Mish for smoother gradients
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, quantum_dim * 2)  # Real and imaginary parts
        )
        
        # Quantum to bio transformation
        self.quantum_to_bio = nn.Sequential(
            nn.Linear(quantum_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, bio_dim)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights for better gradient flow."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='mish')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def bio_to_quantum_state(self, bio_state: torch.Tensor) -> torch.Tensor:
        """Transform biological state to quantum state."""
        # Map to quantum parameters
        quantum_params = self.bio_to_quantum(bio_state)
        
        # Split into real and imaginary components
        real, imag = torch.chunk(quantum_params, 2, dim=-1)
        
        # Create complex quantum state
        quantum_state = torch.complex(real, imag)
        
        # Normalize with careful handling of small values
        norm = torch.norm(quantum_state, dim=-1, keepdim=True)
        norm = torch.where(norm > 1e-8, norm, torch.ones_like(norm))
        quantum_state = quantum_state / norm
        
        return quantum_state
    
    def quantum_to_bio_state(self, quantum_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Transform quantum state to biological state."""
        # Measure quantum state
        measurement, uncertainty = self.measurement.measure_state(quantum_state)
        
        # Transform to biological state
        bio_state = self.quantum_to_bio(measurement)
        
        return bio_state, uncertainty
    
    def forward(self, bio_state: Optional[torch.Tensor] = None,
                quantum_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
        """Process states in either direction."""
        if bio_state is not None:
            quantum_state = self.bio_to_quantum_state(bio_state)
            _, uncertainty = self.measurement.measure_state(quantum_state)
            return quantum_state, uncertainty
            
        elif quantum_state is not None:
            return self.quantum_to_bio_state(quantum_state)
            
        raise ValueError("Either bio_state or quantum_state must be provided")