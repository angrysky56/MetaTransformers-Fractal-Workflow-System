import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data
from modules.stdp.quantum_stdp import QuantumSTDPLayer

def test_quantum_stdp():
    print("Testing Quantum-Enhanced STDP Layer...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data
    num_nodes = 10
    in_channels = 16
    out_channels = 32
    
    # Random input features
    x = torch.randn(num_nodes, in_channels, device=device)
    
    # Create fully connected edge structure
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) 
                             for j in range(num_nodes) if i != j],
                             device=device).t()
    
    print(f"\nCreated test network with:")
    print(f"Nodes: {num_nodes}")
    print(f"Features: {in_channels}")
    print(f"Edge index shape: {edge_index.shape}")
    
    # Initialize STDP layer
    stdp_layer = QuantumSTDPLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        tau_plus=20.0,
        tau_minus=20.0,
        learning_rate=0.01,
        quantum_coupling=0.1
    ).to(device)
    
    print("\nProcessing through STDP layer...")
    
    # Run multiple timesteps to observe STDP and quantum effects
    num_steps = 10
    spikes_history = []
    entanglement_history = []
    
    try:
        for t in range(num_steps):
            # Forward pass
            spikes = stdp_layer(x, edge_index)
            spikes_history.append(spikes.mean().item())
            
            # Compute quantum entanglement
            entanglement = stdp_layer.quantum_entangle()
            entanglement_history.append(entanglement.item())
            
            print(f"\nTimestep {t}:")
            print(f"Average spike rate: {spikes_history[-1]:.4f}")
            print(f"Quantum entanglement: {entanglement_history[-1]:.4f}")
            
            # Update input with some noise to simulate continuous activity
            x = x + 0.1 * torch.randn_like(x)
        
        print("\nFinal Statistics:")
        print(f"Average spike rate across time: {torch.tensor(spikes_history).mean():.4f}")
        print(f"Average quantum entanglement: {torch.tensor(entanglement_history).mean():.4f}")
        
        # Get complex weights for analysis
        weights = stdp_layer.get_complex_weights()
        print(f"\nWeight Statistics:")
        print(f"Mean weight magnitude: {weights.abs().mean().item():.4f}")
        print(f"Mean weight phase: {weights.angle().mean().item():.4f}")
        
        return spikes_history, entanglement_history, weights

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    test_quantum_stdp()