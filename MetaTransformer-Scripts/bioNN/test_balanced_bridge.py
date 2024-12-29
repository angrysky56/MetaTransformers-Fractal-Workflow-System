import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from bioNN.modules.entropy.balanced_entropic_bridge import BalancedBioEntropicBridge

def test_balanced_bridge():
    print("Testing balanced bio-quantum bridge...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Test parameters
    bio_dim = 128
    quantum_dim = 32
    hidden_dim = 96
    batch_size = 10
    num_steps = 10
    
    # Initialize bridge
    bridge = BalancedBioEntropicBridge(
        bio_dim=bio_dim,
        quantum_dim=quantum_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Create test states with varying complexity
    print("\nGenerating test states...")
    test_states = []
    for i in range(3):
        # Create states with different characteristics
        if i == 0:
            # Highly ordered state
            state = torch.zeros(batch_size, bio_dim, device=device)
            state[:, 0] = 1.0  # Single active dimension
        elif i == 1:
            # Mixed state
            state = torch.randn(batch_size, bio_dim, device=device)
            state = F.softmax(state, dim=1)
        else:
            # Highly entropic state
            state = torch.ones(batch_size, bio_dim, device=device)
            state = state / bio_dim
        
        test_states.append(state)
    
    # Test each state type
    for idx, bio_state in enumerate(test_states):
        state_type = ["Ordered", "Mixed", "Entropic"][idx]
        print(f"\nTesting {state_type} State:")
        print("-" * 40)
        
        # Initial conversion
        quantum_state, initial_uncertainty = bridge(bio_state=bio_state)
        print(f"Initial uncertainty: {initial_uncertainty:.6f}")
        
        # Multiple measurement steps
        uncertainties = []
        for step in range(num_steps):
            bio_output, uncertainty = bridge(quantum_state=quantum_state)
            quantum_state, _ = bridge(bio_state=bio_output)
            uncertainties.append(uncertainty)
            
            if step % 2 == 0:
                print(f"Step {step+1} uncertainty: {uncertainty:.6f}")
        
        # Analysis
        avg_uncertainty = sum(uncertainties) / len(uncertainties)
        min_uncertainty = min(uncertainties)
        max_uncertainty = max(uncertainties)
        
        print(f"\nState Analysis:")
        print(f"Average uncertainty: {avg_uncertainty:.6f}")
        print(f"Min uncertainty: {min_uncertainty:.6f}")
        print(f"Max uncertainty: {max_uncertainty:.6f}")
        print(f"Uncertainty range: {max_uncertainty - min_uncertainty:.6f}")
    
    return bridge

if __name__ == "__main__":
    test_balanced_bridge()