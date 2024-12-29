"""
Unified Quantum Bridge
Integrates bio-neural and quantum processing with coherence management and STDP.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import importlib.util
from py2neo import Graph

def import_from_path(module_name: str, file_path: str):
    """Import a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.dirname(os.path.dirname(script_dir))

# Import entropic bridge from path
bridge_path = os.path.join(modules_dir, "entropy", "balanced_entropic_bridge.py")
print(f"Looking for entropy bridge at: {bridge_path}")
if not os.path.exists(bridge_path):
    raise FileNotFoundError(f"Could not find entropy bridge at {bridge_path}")

bridge_module = import_from_path("balanced_entropic_bridge", bridge_path)
BalancedBioEntropicBridge = bridge_module.BalancedBioEntropicBridge

@dataclass
class QuantumBridgeConfig:
    """Configuration for UnifiedQuantumBridge."""
    bio_dim: int = 128
    quantum_dim: int = 64
    hidden_dim: int = 256
    min_coherence: float = 0.85
    growth_threshold: float = 0.92
    adaptation_rate: float = 0.01
    pattern_depth: int = 3
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "00000000"  # Updated to match test config
    quantum_coupling: float = 0.1
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    learning_rate: float = 0.01
    spike_threshold: float = 0.5

class UnifiedQuantumBridge(nn.Module):
    """
    Unified bridge between biological and quantum states with STDP integration.
    """
    def __init__(self, config: Optional[QuantumBridgeConfig] = None):
        super().__init__()
        self.config = config or QuantumBridgeConfig()
        
        # Get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the balanced entropic bridge
        self.bridge = BalancedBioEntropicBridge(
            bio_dim=self.config.bio_dim,
            quantum_dim=self.config.quantum_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        
        # Initialize Neo4j connection
        self.graph = Graph(
            self.config.neo4j_url,
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )
        
        # Initialize complex weights for quantum coupling
        self.complex_weights = nn.Parameter(
            torch.complex(
                torch.randn(self.config.quantum_dim, self.config.quantum_dim) * 0.5,  # Reduced initial magnitude
                torch.randn(self.config.quantum_dim, self.config.quantum_dim) * 0.5
            ).to(self.device)
        )
        
        # Initialize quantum state memory
        self.quantum_memory = torch.zeros(
            self.config.quantum_dim,
            dtype=torch.complex64,
            device=self.device
        )
        
        # Spike and entanglement history
        self.spike_history = []
        self.entanglement_history = []
        self.coherence_history = []
        
        # STDP traces
        self.trace_pre = torch.zeros(self.config.quantum_dim, device=self.device)
        self.trace_post = torch.zeros(self.config.quantum_dim, device=self.device)
        
        # Pattern memory
        self.pattern_memory = []
        
        # Add warmup phase
        self.warmup_steps = 100
        self.current_step = 0
        
        # Adaptive spike threshold
        self.adaptive_threshold = nn.Parameter(
            torch.ones(self.config.quantum_dim, device=self.device) * self.config.spike_threshold
        )    
    def quantum_entangle(self) -> torch.Tensor:
        """
        Compute quantum entanglement of the current state.
        Returns entanglement measure between 0 and 1.
        """
        # Normalize weights
        norm_weights = self.complex_weights / (torch.abs(self.complex_weights).max() + 1e-8)
        
        # Compute density matrix
        density = torch.matmul(norm_weights, norm_weights.conj().T)
        
        # Calculate von Neumann entropy
        eigenvalues = torch.linalg.eigvalsh(density)
        valid_eigs = eigenvalues[eigenvalues > 1e-7]
        if len(valid_eigs) == 0:
            return torch.tensor(0.0, device=self.device)
            
        entropy = -torch.sum(valid_eigs * torch.log2(valid_eigs + 1e-7))
        max_entropy = torch.log2(torch.tensor(len(valid_eigs), dtype=torch.float32))
        
        # Convert entropy to entanglement measure
        entanglement = 1.0 - (entropy / max_entropy)
        return entanglement.clamp(0.0, 1.0)
    
    def update_stdp(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """Update STDP traces and weights."""
        # Convert decay factors to tensors on the correct device
        decay_minus = torch.tensor(-1.0 / self.config.tau_minus, device=self.device)
        decay_plus = torch.tensor(-1.0 / self.config.tau_plus, device=self.device)
    
        # Update traces
        self.trace_pre = self.trace_pre * torch.exp(decay_minus) + pre_spikes
        self.trace_post = self.trace_post * torch.exp(decay_plus) + post_spikes
    
        # Rest of the method remains the same
        pre_contrib = torch.outer(post_spikes, self.trace_pre)
        post_contrib = torch.outer(self.trace_post, pre_spikes)
    
        dw = self.config.learning_rate * (pre_contrib - post_contrib)
    
        phase_update = torch.exp(1j * self.config.quantum_coupling * dw)
        self.complex_weights.data *= phase_update
    
    def measure_coherence(self, quantum_state: torch.Tensor) -> float:
        """Measure quantum coherence of the state."""
        _, uncertainty = self.bridge.measurement.measure_state(quantum_state)
        coherence = 1.0 - uncertainty
        
        # Update history
        self.coherence_history.append(coherence)
        if len(self.coherence_history) > 100:
            self.coherence_history.pop(0)
            
        return coherence
    
    def store_pattern(self, pattern: torch.Tensor, coherence: float):
        """Store quantum pattern in Neo4j if coherence is high enough."""
        if coherence >= self.config.growth_threshold:
            # Get entanglement info
            entanglement = self.quantum_entangle()
            
            # Convert pattern to storable format
            pattern_data = {
                'pattern': pattern.detach().cpu().numpy().tobytes().hex(),
                'coherence': float(coherence),
                'entanglement': float(entanglement),
                'timestamp': torch.cuda.Event.now()
            }

            # Store in Neo4j with entanglement info
            query = """
            MATCH (qb:QuantumBridge {name: 'unified_bridge'})
            CREATE (p:QuantumPattern {
                pattern: $pattern,
                coherence: $coherence,
                entanglement: $entanglement,
                timestamp: datetime(),
                dimension_depth: $dimension_depth,
                has_spikes: true
            })
            CREATE (qb)-[:MAINTAINS_COHERENCE]->(p)
            """
            
            self.graph.run(query, 
                pattern=pattern_data['pattern'],
                coherence=pattern_data['coherence'],
                entanglement=pattern_data['entanglement'],
                dimension_depth=self.config.pattern_depth
            )
            
            # Update local memory
            self.pattern_memory.append(pattern_data)
            if len(self.pattern_memory) > 1000:
                self.pattern_memory.pop(0)

    def forward(self, 
            bio_state: Optional[torch.Tensor] = None,
            quantum_state: Optional[torch.Tensor] = None
           ) -> Tuple[torch.Tensor, float]:
        """
        Forward pass through the quantum bridge.
    
        Args:
            bio_state: Optional biological state tensor to transform to quantum state
            quantum_state: Optional quantum state tensor to transform to biological state
        
        Returns:
            Tuple containing:
                - Transformed state tensor (quantum or biological)
                - Coherence value indicating transformation quality
        """
        # Calculate warmup scale factor
        if self.current_step < self.warmup_steps:
            # Gradually increase scale factor during warmup period
            scale_factor = self.current_step / self.warmup_steps
        else:
            # Use full scale after warmup period
            scale_factor = 1.0
        
        self.current_step += 1

        if bio_state is not None:
            # Bio to quantum transformation
            quantum_state, uncertainty = self.bridge(bio_state=bio_state)
            coherence = 1.0 - uncertainty

            # Adaptive threshold for spikes
            spikes = (torch.abs(quantum_state) > self.adaptive_threshold).float()
            self.spike_history.append(spikes.mean().item())
            
            # Update threshold based on activity
            with torch.no_grad():
                self.adaptive_threshold *= 0.99  # Decay
                self.adaptive_threshold += 0.01 * spikes.mean()  # Adjust based on activity
            
            # Update STDP with previous spikes if available
            if len(self.spike_history) > 1:
                pre_spikes = torch.tensor(self.spike_history[-2], device=self.device)
                post_spikes = spikes
                self.update_stdp(pre_spikes, post_spikes)
            
            # Apply quantum coupling
            quantum_state = torch.matmul(quantum_state, self.complex_weights)
            
            # Measure entanglement
            entanglement = self.quantum_entangle()
            self.entanglement_history.append(entanglement.item())
            
            # Store pattern if coherence is high
            self.store_pattern(quantum_state, coherence)
            
            return quantum_state, coherence
            
        elif quantum_state is not None:
            # Quantum to bio transformation
            bio_state, uncertainty = self.bridge(quantum_state=quantum_state)
            coherence = 1.0 - uncertainty
            
            # Generate spikes from bio state
            spikes = (torch.abs(bio_state) > self.config.spike_threshold).float()
            self.spike_history.append(spikes.mean().item())
            
            quantum_state = bio_state  # For pattern storage
            self.store_pattern(quantum_state, coherence)
            
            return bio_state, coherence
            
        else:
            raise ValueError("Either bio_state or quantum_state must be provided") 
        if self.current_step < self.warmup_steps:
            # Gradually increase scale factor during warmup period
            scale_factor = self.current_step / self.warmup_steps
        else:
            # Use full scale after warmup period
            scale_factor = 1.0
    
        self.current_step += 1

        if bio_state is not None:
            # Bio to quantum transformation
            quantum_state, uncertainty = self.bridge(bio_state=bio_state)
            coherence = 1.0 - uncertainty
    
            # Adaptive threshold for spikes
            spikes = (torch.abs(quantum_state) > self.adaptive_threshold).float()
            self.spike_history.append(spikes.mean().item())
            MAX_HISTORY_LENGTH = 1000
            if len(self.spike_history) > MAX_HISTORY_LENGTH:
                self.spike_history = self.spike_history[-MAX_HISTORY_LENGTH:]
    
            # Update threshold based on activity
            with torch.no_grad():
                self.adaptive_threshold *= 0.99  # Decay
                self.adaptive_threshold += 0.01 * spikes.mean()  # Adjust based on activity
    
            # Update STDP with previous spikes if available
            if len(self.spike_history) > 1:
                pre_spikes = torch.tensor(self.spike_history[-2], device=self.device)
                post_spikes = spikes
                self.update_stdp(pre_spikes, post_spikes)

            # Apply quantum coupling
            try:
                quantum_state = torch.matmul(quantum_state, self.complex_weights)
            except RuntimeError as e:
                logger.error(f"Quantum coupling operation failed: {e}")
                # Handle gracefully or raise custom exception

            # Measure entanglement
            entanglement = self.quantum_entangle()
            self.entanglement_history.append(entanglement.item())

            # Store pattern if coherence is high
            self.store_pattern(quantum_state, coherence)

            return quantum_state, coherence

        elif quantum_state is not None:
            # Quantum to bio transformation
            bio_state, uncertainty = self.bridge(quantum_state=quantum_state)
            coherence = 1.0 - uncertainty

            # Generate spikes from bio state
            spikes = (torch.abs(bio_state) > self.config.spike_threshold).float()
            self.spike_history.append(spikes.mean().item())

            quantum_state = bio_state  # For pattern storage
            self.store_pattern(quantum_state, coherence)

            return bio_state, coherence

        else:
            raise ValueError("Either bio_state or quantum_state must be provided")

    def get_stats(self) -> Dict[str, float]:
        """
        Retrieve operational statistics about the quantum bridge.
    
        Returns:
            Dict[str, float]: A dictionary containing various metrics including:
                - mean_coherence: Average coherence over history
                - mean_spike_rate: Average neural spike rate
                - mean_entanglement: Average quantum entanglement
                - weight_magnitude: Average magnitude of complex weights
                - weight_phase: Average phase of complex weights
                - current_entanglement: Current quantum entanglement value
                - stdp_pre_trace_mean: Average pre-synaptic STDP trace
                - stdp_post_trace_mean: Average post-synaptic STDP trace
                - pattern_count: Total number of stored patterns
                - active_patterns: Number of patterns above minimum coherence
        """
        stats = {
            'mean_coherence': float(np.mean(self.coherence_history)) if self.coherence_history else 0.0,
            'mean_spike_rate': float(np.mean(self.spike_history)) if self.spike_history else 0.0,
            'mean_entanglement': float(np.mean(self.entanglement_history)) if self.entanglement_history else 0.0,
            'weight_magnitude': float(torch.abs(self.complex_weights).mean().item()),
            'weight_phase': float(torch.angle(self.complex_weights).mean().item()),
            'current_entanglement': float(self.quantum_entangle().item()),
            'stdp_pre_trace_mean': float(self.trace_pre.mean().item()),
            'stdp_post_trace_mean': float(self.trace_post.mean().item()),
            'pattern_count': len(self.pattern_memory),
            'active_patterns': sum(1 for p in self.pattern_memory if p['coherence'] > self.config.min_coherence)
        }

        return stats
        if __name__ == "__main__":
            # Simple test
            config = QuantumBridgeConfig()
            bridge = UnifiedQuantumBridge(config)
        
            # Test with random input
            bio_input = torch.randn(128, device=bridge.device)
            quantum_state, coherence = bridge(bio_state=bio_input)
        
            print(f"Test Results:")
            print(f"Coherence: {coherence:.4f}")
            stats = bridge.get_stats()
            for key, value in stats.items():
                print(f"{key}: {value:.4f}")

# At class level:
THRESHOLD_DECAY_RATE = 0.99
THRESHOLD_ADJUSTMENT_RATE = 0.01

def _validate_quantum_state(self, state: torch.Tensor) -> bool:
    return (not torch.isnan(state).any() and 
            not torch.isinf(state).any() and
            state.shape[-1] == self.config.quantum_dim)

    torch.nn.utils.clip_grad_norm_(self.complex_weights, max_norm=1.0)
