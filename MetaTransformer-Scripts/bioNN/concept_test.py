"""
Test feeding quantum entanglement concepts to bioNN
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

# Quantum concept encoding - using quantum properties as features
def create_quantum_concept_data(num_nodes=10, base_features=16):
    # Create feature vectors representing quantum properties
    entanglement_features = torch.zeros(num_nodes, base_features)
    
    # Encode key quantum properties
    entanglement_features[:, 0] = 1.0  # Quantum superposition
    entanglement_features[:, 1] = -1.0  # Classical correlation
    entanglement_features[:, 2] = 0.707  # √(1/2) for equal superposition
    
    # Add phase relationships
    for i in range(num_nodes):
        phase = 2 * np.pi * i / num_nodes
        entanglement_features[i, 3] = np.cos(phase)
        entanglement_features[i, 4] = np.sin(phase)
    
    # Create entangled pair relationships
    edge_list = []
    for i in range(0, num_nodes-1, 2):
        # Connect entangled pairs
        edge_list.extend([[i, i+1], [i+1, i]])
        # Add some classical correlations
        for j in range(num_nodes):
            if j != i and j != i+1:
                edge_list.extend([[i, j], [i+1, j]])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    return entanglement_features, edge_index

def run_concept_test(device='cuda', num_steps=200):
    print("\nInitializing Quantum Concept Test...")
    
    # Create quantum STDP layer with higher quantum coupling
    stdp_layer = QuantumSTDPLayer(
        in_channels=16,
        out_channels=32,
        tau_plus=20.0,
        tau_minus=20.0,
        learning_rate=0.01,
        quantum_coupling=0.3  # Increased quantum influence
    ).to(device)
    
    # Create quantum concept data
    features, edge_index = create_quantum_concept_data()
    features = features.to(device)
    edge_index = edge_index.to(device)
    
    print(f"\nFeature shape: {features.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    
    # Monitor quantum properties
    entanglement_history = []
    coherence_history = []
    spike_history = []
    
    for step in range(num_steps):
        # Process data through STDP
        spikes = stdp_layer(features, edge_index)
        
        # Measure quantum properties
        entanglement = stdp_layer.quantum_entangle()
        coherence = stdp_layer.get_complex_weights().abs().mean()
        
        # Store history
        entanglement_history.append(entanglement.item())
        coherence_history.append(coherence.item())
        spike_history.append(spikes.mean().item())
        
        # Print updates
        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"Entanglement: {entanglement_history[-1]:.4f}")
            print(f"Coherence: {coherence_history[-1]:.4f}")
            print(f"Spike Rate: {spike_history[-1]:.4f}")
        
        # Add small quantum noise to simulate environment
        quantum_noise = torch.randn_like(features) * 0.01
        features = features + quantum_noise
        
    # Analyze results
    print("\nFinal Analysis:")
    print(f"Average Entanglement: {np.mean(entanglement_history):.4f}")
    print(f"Average Coherence: {np.mean(coherence_history):.4f}")
    print(f"Average Spike Rate: {np.mean(spike_history):.4f}")
    
    # Check for learning patterns
    print("\nLearning Patterns:")
    coherence_change = coherence_history[-1] - coherence_history[0]
    print(f"Coherence Change: {coherence_change:.4f}")
    
    spike_pattern = np.array(spike_history)
    pattern_strength = np.std(spike_pattern)
    print(f"Pattern Strength: {pattern_strength:.4f}")
    
    return {
        'entanglement_history': entanglement_history,
        'coherence_history': coherence_history,
        'spike_history': spike_history
    }

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    results = run_concept_test(device=device)
