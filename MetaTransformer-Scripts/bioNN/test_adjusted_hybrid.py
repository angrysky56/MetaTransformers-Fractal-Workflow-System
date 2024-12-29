import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data
from bioNN.modules.hybrid_processor import create_processor
from bioNN.modules.entropy.entropic_bridge import EntropicMeasurement, BioEntropicBridge

def test_adjusted_hybrid():
    print("Starting adjusted hybrid bio-quantum system test...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create sample data with increased dimensionality
    num_nodes = 10
    input_dim = 16
    
    # Random node features with increased variance
    x = torch.randn(num_nodes, input_dim, device=device) * 2.0  # Increased signal strength
    
    # Create edge indices
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) 
                             for j in range(num_nodes) if i != j], 
                             device=device).t()
    
    print(f"Created edge index tensor with shape: {edge_index.shape}")
    
    # Create test data object
    data = Data(x=x, edge_index=edge_index)
    
    # Initialize processor with adjusted parameters
    processor = create_processor(
        input_dim=input_dim,
        bio_hidden_dim=128,  # Increased hidden dimension
        quantum_dim=32,
        bio_layers=4  # Added an extra layer
    )
    
    # Custom entropic bridge with lower uncertainty threshold
    entropic_bridge = BioEntropicBridge(
        bio_dim=128,
        quantum_dim=32,
        hidden_dim=96  # Increased hidden dimension
    ).to(device)
    
    # Replace processor's bridge
    processor.entropic_bridge = entropic_bridge
    
    print("\nProcessing data through adjusted hybrid system...")
    # Process data
    output, metrics = processor.process_step(data.x, data.edge_index)
    
    # Print results
    print("\nProcessing completed successfully!")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    
    # Get quantum state
    quantum_state, uncertainty = processor.get_quantum_state()
    print(f"\nFinal quantum state uncertainty: {uncertainty:.6f}")
    
    print("\nTensor Information:")
    print(f"Output shape: {output.shape}")
    print(f"Output device: {output.device}")
    print(f"Quantum state shape: {quantum_state.shape}")
    
    return output, metrics, quantum_state, uncertainty

if __name__ == "__main__":
    test_adjusted_hybrid()