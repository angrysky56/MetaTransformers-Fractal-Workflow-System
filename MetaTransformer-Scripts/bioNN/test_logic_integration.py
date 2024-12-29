"""
Test BioNN-Logic Integration
"""

import os
import torch
from logic_processing.activate_logic import setup_logic_paths
from logic_processing.quantum_logic_bridge import QuantumLogicBridge

def test_logic_integration():
    print("Testing BioNN-Logic Integration...")
    
    # Setup Logic-LLM paths
    print("\nSetting up Logic-LLM paths...")
    paths = setup_logic_paths()
    for key, value in paths.items():
        print(f"- {key}: {value}")
    
    # Initialize bridge
    print("\nInitializing quantum logic bridge...")
    bridge = QuantumLogicBridge()
    
    # Create test quantum state
    print("\nGenerating test quantum state...")
    state = torch.randn(3, 3, dtype=torch.complex64)
    
    # Test data
    data = {
        "context": "Pattern exhibits quantum coherence through logical structure",
        "metadata": {
            "type": "test_pattern",
            "dimension": 3
        }
    }
    
    print("\nProcessing quantum state through logic bridge...")
    result = bridge.process_quantum_state(state, data)
    
    if result.get("success", False):
        print("\nProcessing successful!")
        print(f"- Coherence: {result['coherence']:.3f}")
        print(f"- New state shape: {result['new_state'].shape}")
        if result.get("inference"):
            print(f"- Inference results: {result['inference']}")
    else:
        print(f"\nProcessing failed: {result.get('error', 'Unknown error')}")
        if "paths_checked" in result:
            print("\nPaths checked:")
            for key, path in result["paths_checked"].items():
                print(f"- {key}: {path}")
        if not bridge.logic_llm_available:
            print("\nNOTE: Logic-LLM integration is not available")
            print("Please check if all required files exist in the Logic-LLM directory")

if __name__ == "__main__":
    test_logic_integration()
