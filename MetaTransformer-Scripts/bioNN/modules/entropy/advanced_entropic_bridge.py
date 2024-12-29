import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class AdaptiveEntropicMeasurement:
    """Enhanced entropic measurement system with adaptive uncertainty reduction."""
    
    def __init__(self, measurement_dim: int, 
                 base_uncertainty_threshold: float = 0.1,
                 eigenvalue_cutoff: float = 1e-6,
                 adaptation_rate: float = 0.1):
        self.measurement_dim = measurement_dim
        self.base_uncertainty_threshold = base_uncertainty_threshold
        self.eigenvalue_cutoff = eigenvalue_cutoff
        self.adaptation_rate = adaptation_rate
        self.measurement_history = []
        
    def adaptive_von_neumann_entropy(self, density_matrix: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Calculate adapted von Neumann entropy with uncertainty normalization."""
        try:
            # Get eigenvalues and sort them
            eigenvalues = torch.linalg.eigvals(density_matrix).real
            eigenvalues = torch.sort(eigenvalues.abs(), descending=True)[0]
            
            # Normalize eigenvalues
            total = torch.sum(eigenvalues)
            if total > 0:
                eigenvalues = eigenvalues / total
            
            # Apply adaptive cutoff
            significant_eigenvalues = eigenvalues[eigenvalues > self.eigenvalue_cutoff]
            
            if len(significant_eigenvalues) > 0:
                # Calculate entropy only for significant eigenvalues
                entropy = -torch.sum(significant_eigenvalues * 
                                   torch.log2(significant_eigenvalues + 1e-10))
                
                # Normalize entropy by the theoretical maximum
                max_entropy = np.log2(len(significant_eigenvalues))
                normalized_uncertainty = float(entropy / max_entropy if max_entropy > 0 else 0.0)
                
                # Apply adaptive smoothing based on history
                if self.measurement_history:
                    avg_uncertainty = sum(self.measurement_history) / len(self.measurement_history)
                    normalized_uncertainty = (avg_uncertainty * (1 - self.adaptation_rate) + 
                                           normalized_uncertainty * self.adaptation_rate)
                
                # Update history
                self.measurement_history.append(normalized_uncertainty)
                if len(self.measurement_history) > 10:  # Keep last 10 measurements
                    self.measurement_history.pop(0)
                
                return entropy, normalized_uncertainty
            
        except Exception as e:
            print(f"Error in entropy calculation: {e}")
        
        # Fallback values
        return torch.tensor(0.0), 1.0
    
    def measure_state(self, quantum_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Perform measurement with enhanced uncertainty quantification."""
        # Create density matrix
        density_matrix = torch.outer(quantum_state.flatten(), 
                                   quantum_state.flatten().conj())
        
        # Calculate entropy and uncertainty
        _, uncertainty = self.adaptive_von_neumann_entropy(density_matrix)
        
        # Calculate measurement outcome
        measurement = torch.abs(quantum_state) ** 2
        
        return measurement, uncertainty

class EnhancedBioEntropicBridge(nn.Module):
    """Enhanced bridge between quantum and biological neural processing."""
    
    def __init__(self, bio_dim: int, quantum_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.bio_dim = bio_dim
        self.quantum_dim = quantum_dim
        
        # Enhanced entropic measurement system
        self.measurement = AdaptiveEntropicMeasurement(
            measurement_dim=quantum_dim,
            base_uncertainty_threshold=0.1,
            eigenvalue_cutoff=1e-6,
            adaptation_rate=0.1
        )
        
        # Enhanced quantum to biological mapping
        self.quantum_to_bio = nn.Sequential(
            nn.Linear(quantum_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Add normalization
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim, bio_dim)
        )
        
        # Enhanced biological to quantum mapping
        self.bio_to_quantum = nn.Sequential(
            nn.Linear(bio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, quantum_dim * 2)  # Real and imaginary parts
        )
        
    def bio_to_quantum_state(self, bio_state: torch.Tensor) -> torch.Tensor:
        """Convert biological neural state to quantum state with enhanced normalization."""
        # Map to quantum dimensions
        quantum_params = self.bio_to_quantum(bio_state)
        
        # Split into real and imaginary parts
        real, imag = torch.chunk(quantum_params, 2, dim=-1)
        
        # Create complex quantum state with improved normalization
        quantum_state = torch.complex(real, imag)
        quantum_state = F.normalize(quantum_state, p=2, dim=-1)
        
        return quantum_state
    
    def quantum_to_bio_state(self, quantum_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Convert quantum state to biological neural state with enhanced uncertainty."""
        # Measure quantum state
        measurement, uncertainty = self.measurement.measure_state(quantum_state)
        
        # Map to biological dimensions
        bio_state = self.quantum_to_bio(measurement)
        
        return bio_state, uncertainty
    
    def forward(self, bio_state: Optional[torch.Tensor] = None, 
                quantum_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
        """Enhanced bridging between quantum and biological domains."""
        if bio_state is not None:
            # Biological to quantum
            quantum_state = self.bio_to_quantum_state(bio_state)
            # Get initial uncertainty measurement
            _, uncertainty = self.measurement.measure_state(quantum_state)
            return quantum_state, uncertainty
        
        elif quantum_state is not None:
            # Quantum to biological
            return self.quantum_to_bio_state(quantum_state)
        
        else:
            raise ValueError("Either bio_state or quantum_state must be provided")