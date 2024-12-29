import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from bioNN.modules.entropy.advanced_entropic_bridge import EnhancedBioEntropicBridge

def test_enhanced_bridge():
    print("Testing enhanced bio-quantum bridge...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Parameters
    bio_dim = 128
    quantum_dim = 32
    hidden_dim = 96
    batch_size = 10
    
    # Create enhanced bridge
    bridge = EnhancedBioEntropicBridge(
        bio_dim=bio_dim,
        quantum_dim=quantum_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Create test biological state
    bio_state = torch.randn(batch_size, bio_dim, device=device)
    bio_state = F.normalize(bio_state, p=2, dim=1)  # Normalize input
    
    print("\nProcessing biological to quantum state...")
    # Convert to quantum state
    quantum_state, initial_uncertainty = bridge(bio_state=bio_state)
    
    print(f"Initial uncertainty: {initial_uncertainty:.6f}")
    print(f"Quantum state norm: {torch.norm(quantum_state).item():.6f}")
    
    # Multiple measurement steps to test uncertainty reduction
    print("\nPerforming multiple measurements...")
    uncertainties = []
    for i in range(5):
        bio_output, uncertainty = bridge(quantum_state=quantum_state)
        quantum_state, new_uncertainty = bridge(bio_state=bio_output)
        uncertainties.append(uncertainty)
        print(f"Step {i+1} uncertainty: {uncertainty:.6f}")
    
    print("\nFinal state analysis:")
    print(f"Final quantum state shape: {quantum_state.shape}")
    print(f"Average uncertainty: {sum(uncertainties)/len(uncertainties):.6f}")
    print(f"Minimum uncertainty achieved: {min(uncertainties):.6f}")
    
    return quantum_state, uncertainties

if __name__ == "__main__":
    test_enhanced_bridge()