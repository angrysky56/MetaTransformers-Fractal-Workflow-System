"""
Test suite for UnifiedQuantumBridge.
Tests quantum-bio transitions, STDP, and pattern formation.
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from bioNN.modules.quantum_integration.unified_quantum_bridge import (
    UnifiedQuantumBridge,
    QuantumBridgeConfig
)

def test_unified_bridge(
    num_steps: int = 10,
    bio_dim: int = 128,
    quantum_dim: int = 64,
    batch_size: int = 8
):
    """
    Test the UnifiedQuantumBridge with STDP and quantum coupling.
    """
    print("\nInitializing Unified Quantum Bridge Test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize with test configuration
    config = QuantumBridgeConfig(
        bio_dim=bio_dim,
        quantum_dim=quantum_dim,
        hidden_dim=256,
        min_coherence=0.85,
        growth_threshold=0.90,
        quantum_coupling=0.1,
        tau_plus=20.0,
        tau_minus=20.0,
        learning_rate=0.01,
        spike_threshold=0.5
    )
    
    bridge = UnifiedQuantumBridge(config).to(device)
    print("\nBridge initialized successfully")
    
    print(f"\nRunning {num_steps} timesteps with batch size {batch_size}...")
    
    # Track metrics over time
    metrics_history = []
    coherence_values = []
    entanglement_values = []
    spike_rates = []
    
    try:
        for step in range(num_steps):
            # Generate batch of bio inputs with temporal correlation
            base_input = torch.randn(batch_size, bio_dim, device=device)
            temporal_noise = 0.1 * torch.randn(batch_size, bio_dim, device=device)
            bio_input = base_input + temporal_noise
            
            # Process through bridge
            quantum_state, coherence = bridge(bio_state=bio_input)
            
            # Get current stats
            stats = bridge.get_stats()
            metrics_history.append(stats)
            
            # Track key metrics
            coherence_values.append(coherence)
            entanglement_values.append(stats['mean_entanglement'])
            spike_rates.append(stats['mean_spike_rate'])
            
            print(f"\nTimestep {step}:")
            print(f"Coherence: {coherence:.4f}")
            print(f"Entanglement: {stats['mean_entanglement']:.4f}")
            print(f"Spike Rate: {stats['mean_spike_rate']:.4f}")
            print(f"Weight Stats:")
            print(f"- Magnitude: {stats['weight_magnitude']:.4f}")
            print(f"- Phase: {stats['weight_phase']:.4f}")
            
            # Test quantum to bio conversion
            bio_reconstruction, _ = bridge(quantum_state=quantum_state)
            reconstruction_error = torch.nn.functional.mse_loss(
                bio_reconstruction, bio_input
            )
            print(f"Reconstruction Error: {reconstruction_error.item():.6f}")
            
        # Print final summary
        print("\nFinal Statistics:")
        print(f"Average Metrics Across Time:")
        print(f"- Coherence: {np.mean(coherence_values):.4f} ± {np.std(coherence_values):.4f}")
        print(f"- Entanglement: {np.mean(entanglement_values):.4f} ± {np.std(entanglement_values):.4f}")
        print(f"- Spike Rate: {np.mean(spike_rates):.4f} ± {np.std(spike_rates):.4f}")
        
        # Load and verify stored patterns
        patterns = bridge.load_patterns(min_coherence=0.90)
        print(f"\nStored Patterns:")
        print(f"Total patterns: {len(patterns)}")
        if patterns:
            coherence_values = [p['coherence'] for p in patterns]
            entanglement_values = [p['entanglement'] for p in patterns]
            print(f"Pattern Statistics:")
            print(f"- Mean Coherence: {np.mean(coherence_values):.4f}")
            print(f"- Mean Entanglement: {np.mean(entanglement_values):.4f}")
        
        return True, metrics_history
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, metrics_history

def plot_metrics(metrics_history: List[Dict[str, float]]):
    """Plot the evolution of key metrics."""
    try:
        import matplotlib.pyplot as plt
        
        # Extract metrics
        steps = range(len(metrics_history))
        coherence = [m['mean_coherence'] for m in metrics_history]
        entanglement = [m['mean_entanglement'] for m in metrics_history]
        spike_rate = [m['mean_spike_rate'] for m in metrics_history]
        weight_mag = [m['weight_magnitude'] for m in metrics_history]
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('UnifiedQuantumBridge Metrics Evolution')
        
        # Coherence
        axes[0,0].plot(steps, coherence, 'b-', label='Coherence')
        axes[0,0].set_title('Mean Coherence')
        axes[0,0].set_xlabel('Step')
        axes[0,0].set_ylabel('Coherence')
        axes[0,0].grid(True)
        
        # Entanglement
        axes[0,1].plot(steps, entanglement, 'r-', label='Entanglement')
        axes[0,1].set_title('Mean Entanglement')
        axes[0,1].set_xlabel('Step')
        axes[0,1].set_ylabel('Entanglement')
        axes[0,1].grid(True)
        
        # Spike Rate
        axes[1,0].plot(steps, spike_rate, 'g-', label='Spike Rate')
        axes[1,0].set_title('Mean Spike Rate')
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('Rate')
        axes[1,0].grid(True)
        
        # Weight Magnitude
        axes[1,1].plot(steps, weight_mag, 'm-', label='Weight Magnitude')
        axes[1,1].set_title('Mean Weight Magnitude')
        axes[1,1].set_xlabel('Step')
        axes[1,1].set_ylabel('Magnitude')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
        pass

if __name__ == "__main__":
    # Run test
    success, metrics_history = test_unified_bridge(
        num_steps=20,
        bio_dim=128,
        quantum_dim=64,
        batch_size=8
    )
    
    if success and metrics_history:
        # Plot results
        plot_metrics(metrics_history)
