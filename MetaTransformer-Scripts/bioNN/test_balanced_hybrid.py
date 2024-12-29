import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data
from bioNN.modules.hybrid_processor_balanced import create_balanced_processor

def test_balanced_hybrid():
    print("Starting balanced hybrid bio-quantum system test...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Test parameters
    num_nodes = 10
    input_dim = 16
    bio_hidden_dim = 128
    quantum_dim = 32
    num_steps = 5
    
    print(f"\nInitializing with:")
    print(f"- Nodes: {num_nodes}")
    print(f"- Input dimensions: {input_dim}")
    print(f"- Bio hidden dimensions: {bio_hidden_dim}")
    print(f"- Quantum dimensions: {quantum_dim}")
    
    # Create test data
    x = torch.randn(num_nodes, input_dim, device=device)
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) 
                             for j in range(num_nodes) if i != j], 
                             device=device).t()
    
    data = Data(x=x, edge_index=edge_index)
    
    # Initialize balanced processor
    processor = create_balanced_processor(
        input_dim=input_dim,
        bio_hidden_dim=bio_hidden_dim,
        quantum_dim=quantum_dim,
        bio_layers=4
    )
    
    print("\nRunning processing steps...")
    # Process multiple steps
    for step in range(num_steps):
        print(f"\nStep {step + 1}:")
        print("-" * 40)
        
        output, metrics = processor.process_step(data.x, data.edge_index)
        
        # Print metrics
        print("Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
                
        # Get system state
        quantum_state, uncertainty = processor.get_quantum_state()
        print(f"\nQuantum state shape: {quantum_state.shape}")
        print(f"Current uncertainty: {uncertainty:.6f}")
        
        # Get comprehensive metrics periodically
        if step % 2 == 0:
            print("\nSystem Metrics:")
            system_metrics = processor.get_system_metrics()
            for key, value in system_metrics.items():
                print(f"  {key}: {value:.6f}")
    
    print("\nFinal System Status:")
    print("-" * 40)
    final_metrics = processor.get_system_metrics()
    for key, value in final_metrics.items():
        print(f"{key}: {value:.6f}")
    
    return processor, final_metrics

if __name__ == "__main__":
    test_balanced_hybrid()