"""
Test using BalancedEntropicBridge for better coherence management
"""
import os
import sys
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import torch.nn.functional as F

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from bioNN.modules.stdp import QuantumSTDPLayer
from bioNN.entropy.balanced_entropic_bridge import BalancedBioEntropicBridge, EntropyConfig

class BalancedQuantumProcessor:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Align dimensions between STDP and bridge
        self.input_dim = 32
        self.hidden_dim = 64
        self.quantum_dim = 32
        
        # Initialize STDP with aligned dimensions
        self.stdp_layer = QuantumSTDPLayer(
            in_channels=self.input_dim,
            out_channels=self.quantum_dim,
            tau_plus=20.0,
            tau_minus=20.0,
            learning_rate=0.01,
            quantum_coupling=0.3
        ).to(device)
        
        # Initialize balanced entropic bridge
        self.bridge = BalancedBioEntropicBridge(
            bio_dim=self.input_dim,
            quantum_dim=self.quantum_dim,
            hidden_dim=self.hidden_dim
        ).to(device)
        
        self.entropy_config = EntropyConfig()
        
    def create_test_data(self, num_nodes=10):
        # Create biological features matching input dimension
        bio_features = torch.randn(num_nodes, self.input_dim).to(self.device)
        bio_features = F.normalize(bio_features, dim=1)
        
        # Create quantum state through bridge
        quantum_state, _ = self.bridge(bio_features)
        
        # Create edge connections
        edge_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_list.append([i, j])
        edge_index = torch.tensor(edge_list, dtype=torch.long).to(self.device).t()
        
        return bio_features, quantum_state.abs(), edge_index  # Use amplitude only

def run_balanced_test(num_steps=200):
    print("\nInitializing Balanced Quantum Test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    processor = BalancedQuantumProcessor(device)
    
    # Create initial states
    bio_state, quantum_state, edge_index = processor.create_test_data()
    
    print(f"\nState dimensions:")
    print(f"Bio state: {bio_state.shape}")
    print(f"Quantum state: {quantum_state.shape}")
    print(f"Edge index: {edge_index.shape}")
    
    # Track metrics
    metrics = {
        'entanglement': [],
        'coherence': [],
        'uncertainty': [],
        'spikes': []
    }
    
    print("\nRunning test with balanced entropy management...")
    for step in range(num_steps):
        try:
            # Process through STDP
            spikes = processor.stdp_layer(quantum_state, edge_index)
            
            # Update quantum state through bridge
            bio_state, uncertainty = processor.bridge(quantum_state=quantum_state)
            quantum_state_complex, _ = processor.bridge(bio_state=bio_state)
            quantum_state = quantum_state_complex.abs()  # Use amplitude
            
            # Measure quantum properties
            entanglement = processor.stdp_layer.quantum_entangle()
            coherence = processor.stdp_layer.get_complex_weights().abs().mean()
            
            # Store metrics
            metrics['entanglement'].append(entanglement.item())
            metrics['coherence'].append(coherence.item())
            metrics['uncertainty'].append(uncertainty)
            metrics['spikes'].append(spikes.mean().item())
            
            # Print updates
            if step % 20 == 0:
                print(f"\nStep {step}:")
                print(f"Entanglement: {entanglement.item():.4f}")
                print(f"Coherence: {coherence.item():.4f}")
                print(f"Uncertainty: {uncertainty:.4f}")
                print(f"Spike Rate: {spikes.mean().item():.4f}")
                
            # Add small quantum noise for evolution
            noise = torch.randn_like(quantum_state) * 0.01
            quantum_state = F.normalize(quantum_state + noise, dim=-1)
            
        except Exception as e:
            print(f"\nError at step {step}: {str(e)}")
            print("Continuing...")
            continue
    
    # Final analysis
    print("\nFinal Metrics:")
    for key, values in metrics.items():
        if values:  # Check if we have any values
            avg_value = np.mean(values)
            final_value = values[-1]
            change = final_value - values[0]
            print(f"\n{key.capitalize()}:")
            print(f"  Average: {avg_value:.4f}")
            print(f"  Final: {final_value:.4f}")
            print(f"  Change: {change:.4f}")
    
    return metrics

if __name__ == "__main__":
    metrics = run_balanced_test()