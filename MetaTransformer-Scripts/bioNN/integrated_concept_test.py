"""
Test feeding concepts through proper bioNN quantum bridge configuration
"""
import os
import sys
from pathlib import Path
import torch
from datetime import datetime
import numpy as np

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from bioNN.modules.stdp import QuantumSTDPLayer
from ai_ml_lab.quantum_monitor import QuantumMonitor

class IntegratedQuantumProcessor:
    def __init__(self, device='cuda'):
        self.device = device
        self.attention_temp = 5.0
        self.coherence_target = 0.95
        self.stability_target = 0.9
        
        # Initialize STDP with proper bridge configuration
        self.stdp_layer = QuantumSTDPLayer(
            in_channels=16,
            out_channels=32,
            tau_plus=20.0,
            tau_minus=20.0,
            learning_rate=0.01,
            quantum_coupling=0.5  # Higher coupling for quantum-logic hybrid
        ).to(device)
        
    def create_multi_channel_features(self, num_nodes=10):
        # Create three channel types as specified in bridge
        quantum_state = torch.randn(num_nodes, 16).to(self.device) * 0.1
        bio_state = torch.zeros(num_nodes, 16).to(self.device)
        hybrid_state = torch.zeros(num_nodes, 16).to(self.device)
        
        # Initialize biological patterns
        for i in range(num_nodes):
            # Biological firing pattern
            bio_state[i, i % 4] = 1.0
            # Hybrid state combines both
            hybrid_state[i] = (quantum_state[i] + bio_state[i]) / 2
            
        return {
            'quantum': quantum_state,
            'bio': bio_state,
            'hybrid': hybrid_state
        }
        
    def create_adaptive_connections(self, num_nodes):
        # Create adaptive connection pattern
        edge_list = []
        
        # Initial biological connections
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_list.append([i, j])
                    
        # Add quantum entanglement pairs
        for i in range(0, num_nodes-1, 2):
            edge_list.extend([[i, i+1], [i+1, i]])
            
        edge_index = torch.tensor(edge_list, dtype=torch.long).to(self.device).t()
        return edge_index
        
    def process_with_attention(self, states, edge_index):
        # Apply attention mechanism
        attention_weights = torch.softmax(
            torch.randn(3) * self.attention_temp, 
            dim=0
        ).to(self.device)
        
        # Combine states with attention
        combined_state = (
            states['quantum'] * attention_weights[0] +
            states['bio'] * attention_weights[1] + 
            states['hybrid'] * attention_weights[2]
        )
        
        return self.stdp_layer(combined_state, edge_index)

def run_integrated_test(num_steps=200):
    print("\nInitializing Integrated Quantum-Bio Test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    processor = IntegratedQuantumProcessor(device)
    
    # Initialize states and connections
    states = processor.create_multi_channel_features()
    edge_index = processor.create_adaptive_connections(10)
    
    # Track metrics
    metrics = {
        'entanglement': [],
        'coherence': [],
        'spikes': [],
        'attention': []
    }
    
    print("\nRunning integration test...")
    for step in range(num_steps):
        # Process through attention mechanism
        spikes = processor.process_with_attention(states, edge_index)
        
        # Measure quantum properties
        entanglement = processor.stdp_layer.quantum_entangle()
        coherence = processor.stdp_layer.get_complex_weights().abs().mean()
        
        # Store metrics
        metrics['entanglement'].append(entanglement.item())
        metrics['coherence'].append(coherence.item())
        metrics['spikes'].append(spikes.mean().item())
        
        # Print progress
        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"Entanglement: {entanglement.item():.4f}")
            print(f"Coherence: {coherence.item():.4f}")
            print(f"Spike Rate: {spikes.mean().item():.4f}")
            
        # Adaptive channel mixing
        if step % 10 == 0:
            # Update quantum state based on biological activity
            states['quantum'] = states['quantum'] * 0.9 + states['bio'] * 0.1
            # Update hybrid state
            states['hybrid'] = (states['quantum'] + states['bio']) / 2
            
    # Final analysis
    print("\nFinal Metrics:")
    for key, values in metrics.items():
        print(f"{key.capitalize()}:")
        print(f"  Average: {np.mean(values):.4f}")
        print(f"  Final: {values[-1]:.4f}")
        print(f"  Change: {values[-1] - values[0]:.4f}")
        
    return metrics

if __name__ == "__main__":
    run_integrated_test()