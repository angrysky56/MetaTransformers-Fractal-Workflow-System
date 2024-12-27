import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
# Add the parent directory to system path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data
from bioNN.modules.hybrid_processor import create_processor

def test_hybrid_system():
    print("Starting hybrid bio-quantum system test...")
    
    # Enable CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create sample data
    num_nodes = 10
    input_dim = 16
    
    print(f"\nInitializing test data with {num_nodes} nodes and {input_dim} features...")
    
    # Random node features
    x = torch.randn(num_nodes, input_dim, device=device)
    
    # Sample edge indices (fully connected for testing)
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) 
                             for j in range(num_nodes) if i != j], 
                             device=device).t()
    
    print(f"Created edge index tensor with shape: {edge_index.shape}")
    
    # Create test data object
    data = Data(x=x, edge_index=edge_index)
    
    print("\nInitializing hybrid processor...")
    # Initialize processor
    processor = create_processor(
        input_dim=input_dim,
        bio_hidden_dim=64,
        quantum_dim=32,
        bio_layers=3
    )
    
    print("\nProcessing data through hybrid system...")
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
    
    # Additional info
    print("\nTensor Information:")
    print(f"Output shape: {output.shape}")
    print(f"Output device: {output.device}")
    print(f"Quantum state shape: {quantum_state.shape}")
    
    return output, metrics, quantum_state, uncertainty

if __name__ == "__main__":
    test_hybrid_system()