"""
Activate Logic-LLM Integration
"""

import os
import sys
from pathlib import Path

def setup_logic_paths():
    """Setup paths for Logic-LLM integration"""
    # Get base paths
    base_path = Path(__file__).parent
    logic_llm_path = base_path / "Logic-LLM"
    
    # Add all necessary paths
    paths_to_add = [
        str(base_path),
        str(logic_llm_path),
        str(logic_llm_path / "models"),
        str(logic_llm_path / "models/prompts"),
        str(logic_llm_path / "models/symbolic_solvers")
    ]
    
    # Add to sys.path if not already there
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.append(path)
            
    return {
        "base_path": str(base_path),
        "logic_llm_path": str(logic_llm_path),
        "paths_added": paths_to_add
    }

if __name__ == "__main__":
    paths = setup_logic_paths()
    print("\nLogic-LLM paths setup:")
    for key, value in paths.items():
        print(f"{key}: {value}")
