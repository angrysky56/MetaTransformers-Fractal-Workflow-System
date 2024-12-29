"""
Logic Processing Integration for BioNN
"""

import os
import sys
from pathlib import Path

# Update paths for new location
BIONN_PATH = Path(__file__).parent.parent
LOGIC_LLM_PATH = Path(__file__).parent / "Logic-LLM"
LOGIC_PROCESSING_PATH = Path(__file__).parent / "logic-processing"

# Add all required paths
sys.path.append(str(BIONN_PATH))
sys.path.append(str(LOGIC_LLM_PATH))
sys.path.append(str(LOGIC_PROCESSING_PATH))

# Export paths for other modules
__all__ = ['BIONN_PATH', 'LOGIC_LLM_PATH', 'LOGIC_PROCESSING_PATH']
