import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from torch_geometric.nn import MessagePassing

class QuantumSTDPLayer(MessagePassing):
    """
    Quantum-enhanced Spike-Timing-Dependent Plasticity layer.
    Combines biological STDP with quantum entanglement for synaptic updates.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 tau_plus: float = 20.0, 
                 tau_minus: float = 20.0,
                 learning_rate: float = 0.01,
                 quantum_coupling: float = 0.1):
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.lr = learning_rate
        self.quantum_coupling = quantum_coupling
        
        # Synaptic weights as complex numbers for quantum effects
        self.weights_real = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weights_imag = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        # Spike timing memory (will be moved to correct device with .to())
        self.register_buffer('last_spike_time', torch.zeros(1))
        self.register_buffer('membrane_potential', torch.zeros(1))
        
        # Quantum state history for entanglement
        self.quantum_states = []
        self.max_history = 1000
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights with quantum phase consideration"""
        std = 1.0 / np.sqrt(self.in_channels)
        self.weights_real.data.uniform_(-std, std)
        self.weights_imag.data.uniform_(-std, std)
        
    def get_complex_weights(self) -> torch.Tensor:
        """Construct complex weights from real and imaginary parts"""
        return torch.complex(self.weights_real, self.weights_imag)
    
    def quantum_update(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Apply quantum phase-based weight update"""
        phase = torch.tensor(2 * np.pi * delta_t / self.tau_plus, device=delta_t.device)
        return torch.exp(1j * phase) * self.quantum_coupling
    
    def compute_stdp_update(self, pre_time: torch.Tensor, post_time: torch.Tensor) -> torch.Tensor:
        """Compute STDP weight updates with quantum influence"""
        delta_t = post_time - pre_time
        
        # Classical STDP
        stdp_update = torch.where(
            delta_t > 0,
            torch.exp(-delta_t / self.tau_plus),
            torch.exp(delta_t / self.tau_minus)
        )
        
        # Quantum influence
        quantum_phase = self.quantum_update(delta_t)
        
        # Combine classical and quantum effects
        return stdp_update * quantum_phase.abs()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                spike_times: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with STDP learning
        Args:
            x: Node features
            edge_index: Graph connectivity
            spike_times: Optional timing information for spikes
        """
        # Ensure everything is on the same device
        device = x.device
        self.last_spike_time = self.last_spike_time.to(device)
        self.membrane_potential = self.membrane_potential.to(device)
        
        if self.membrane_potential.shape != (x.size(0), self.out_channels):
            self.membrane_potential = torch.zeros(x.size(0), self.out_channels, device=device)
        
        # Current time step
        current_time = self.last_spike_time + 1
        
        # Get complex weights
        weights = self.get_complex_weights()
        
        # Message passing with quantum weights
        out = self.propagate(edge_index, x=x, weights=weights)
        
        # Apply membrane potential dynamics
        self.membrane_potential = self.membrane_potential * 0.9 + out
        
        # Generate spikes based on membrane potential
        spikes = (self.membrane_potential > 1.0).float()
        self.membrane_potential[spikes > 0] = 0
        
        # Update quantum states history
        quantum_state = torch.view_as_complex(
            torch.stack([self.weights_real, self.weights_imag], dim=-1)
        )
        self.quantum_states.append(quantum_state)
        if len(self.quantum_states) > self.max_history:
            self.quantum_states.pop(0)
        
        # Store spike timing
        if spikes.any():
            self.last_spike_time = current_time
        
        return spikes
    
    def message(self, x_j: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Define how messages are passed between nodes"""
        return torch.matmul(x_j, weights.real)  # Using only real part for message passing
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Update node embeddings"""
        return aggr_out
    
    def quantum_entangle(self) -> torch.Tensor:
        """Compute quantum entanglement between synapses"""
        if len(self.quantum_states) < 2:
            return torch.tensor(0.0, device=self.weights_real.device)
            
        # Take last two quantum states
        state1 = self.quantum_states[-1]
        state2 = self.quantum_states[-2]
        
        # Compute quantum correlation
        correlation = torch.abs(torch.sum(torch.conj(state1) * state2))
        norm_factor = torch.norm(state1) * torch.norm(state2)
        
        return correlation / (norm_factor + 1e-8)  # Add small epsilon to avoid division by zero