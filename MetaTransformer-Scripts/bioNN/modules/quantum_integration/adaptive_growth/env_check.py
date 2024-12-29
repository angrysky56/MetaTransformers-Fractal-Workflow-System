"""
Environment verification for adaptive growth system.
"""

import os
import sys
import torch
from typing import Tuple, Dict
from py2neo import Graph
import numpy as np
from pathlib import Path

def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability and version."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    try:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        
        return True, f"CUDA {cuda_version} available with {device_count} device(s): {device_name}"
    except Exception as e:
        return False, f"CUDA error: {str(e)}"

def check_neo4j() -> Tuple[bool, str]:
    """Check Neo4j connection and quantum bridge nodes."""
    try:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "00000000"))
        
        # Check for QuantumBridge node
        result = graph.run("""
            MATCH (qb:QuantumBridge {name: 'unified_bridge'})
            RETURN qb
        """).data()
        
        if not result:
            # Create QuantumBridge node if it doesn't exist
            graph.run("""
                CREATE (qb:QuantumBridge {
                    name: 'unified_bridge',
                    coherence_threshold: 0.85,
                    dimension_depth: 3,
                    bridge_id: 'unified_main',
                    created_at: datetime()
                })
            """)
            return True, "Neo4j connected, created QuantumBridge node"
        
        return True, "Neo4j connected, QuantumBridge node exists"
    except Exception as e:
        return False, f"Neo4j error: {str(e)}"

def check_directories() -> Tuple[bool, str]:
    """Check required directories and files exist."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    
    required_paths = [
        project_root / "modules" / "quantum_integration",
        project_root / "entropy",
        project_root / "modules" / "quantum_integration" / "adaptive_growth"
    ]
    
    missing = []
    for path in required_paths:
        if not path.exists():
            missing.append(str(path))
    
    if missing:
        return False, f"Missing directories: {', '.join(missing)}"
    
    return True, "All required directories present"

def verify_environment() -> Tuple[bool, Dict[str, str]]:
    """
    Verify all required components for the adaptive growth system.
    Returns (success, status_dict).
    """
    status = {}
    
    # Check CUDA
    cuda_ok, cuda_status = check_cuda()
    status['cuda'] = cuda_status
    
    # Check Neo4j
    neo4j_ok, neo4j_status = check_neo4j()
    status['neo4j'] = neo4j_status
    
    # Check directories
    dirs_ok, dirs_status = check_directories()
    status['directories'] = dirs_status
    
    # Overall success requires all checks to pass
    success = all([cuda_ok, neo4j_ok, dirs_ok])
    
    return success, status

if __name__ == "__main__":
    success, status = verify_environment()
    print(f"\nEnvironment verification {'succeeded' if success else 'failed'}:")
    for key, value in status.items():
        print(f"{key}: {value}")
