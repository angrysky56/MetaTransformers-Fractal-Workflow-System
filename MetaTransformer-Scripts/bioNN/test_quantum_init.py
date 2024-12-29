"""
Basic initialization test for unified quantum bridge.
"""

import os
import sys
import torch

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(parent_dir))

def test_imports():
    print("\nTesting imports...")
    
    try:
        from bioNN.modules.quantum_integration.unified_quantum_bridge import (
            UnifiedQuantumBridge, 
            UnifiedQuantumConfig
        )
        print("Successfully imported unified quantum bridge")
        return True
    except Exception as e:
        print(f"Import error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_initialization():
    print("\nTesting bridge initialization...")
    
    from bioNN.modules.quantum_integration.unified_quantum_bridge import (
        UnifiedQuantumBridge, 
        UnifiedQuantumConfig
    )
    
    try:
        config = UnifiedQuantumConfig()
        print("\nCreated configuration:")
        for key, value in config.__dict__.items():
            print(f"- {key}: {value}")
            
        bridge = UnifiedQuantumBridge(config)
        print("\nSuccessfully created bridge instance")
        return True
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running quantum bridge initialization tests...")
    
    if test_imports():
        test_initialization()