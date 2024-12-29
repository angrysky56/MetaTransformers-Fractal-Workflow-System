# Fractal Logic System Setup Guide

## Overview
This system integrates logical reasoning capabilities with fractal pattern evolution. It can be used either:
1. As a standalone system using our simplified logical core
2. With the full Logic-LLM integration for more advanced capabilities

## Dependencies
### Required:
- Python 3.10+  
- Neo4j database
- OpenAI API key (for GPT-4 integration)

### Optional:
- Logic-LLM repository (for advanced symbolic reasoning)

## Installation

### 1. Basic Setup (Standalone Mode)
```bash
# Create conda environment
conda create -n logic-meta python=3.10
conda activate logic-meta

# Install requirements
pip install -r requirements.txt

# Initialize Neo4j database (if not already running)
# Default credentials in config.yaml
```

### 2. Full Setup (with Logic-LLM)
```bash
# Clone Logic-LLM (if using advanced features)
git clone https://github.com/Yale-LILY/Logic-LLM.git F:\Logic-LLM

# Install additional requirements
pip install -r F:\Logic-LLM\requirements.txt
```

## Configuration

1. Copy example config:
```bash
cp config.example.yaml config.yaml
```

2. Set required environment variables:
```bash
OPENAI_API_KEY=your_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## For New AI Instances

1. First, search for the Portal information in Neo4j:
```cypher
MATCH (p:IndexPortal)-[:DEFINES_LOCATION]->(sl:ScriptLocation)
RETURN p, sl
```

2. Initialize quantum bridge connection:
```python
from integration.quantum_bridge.connector import QuantumBridge
bridge = QuantumBridge()
bridge.connect()
```

3. Load core processors:
```python
from core.translator.nl2logic import NL2LogicTranslator
from core.solver.symbolic_solver import SymbolicSolver
from core.refiner.self_refinement import SelfRefinement
```

## Usage Examples

### Basic Logical Processing
```python
from fractal_logic_processor import FractalLogicProcessor

async def process_logic():
    processor = FractalLogicProcessor()
    
    result = await processor.process_pattern({
        "context": "Given patterns evolve through quantum fields",
        "query": "How does this pattern evolve?",
        "pattern_state": {
            "tensor": np.eye(3) * 0.8,
            "metadata": {"type": "quantum_logical"}
        }
    })
    
    print(f"Processing result: {result}")
```

### Integration with Existing Systems
```python
# Connect to autonomous pattern system
from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=(neo4j_user, neo4j_password))

# Query existing pattern system
with driver.session() as session:
    result = session.run("""
        MATCH (aps:AutonomousPatternSystem {name: 'FractalNetworkController'})
        RETURN aps
    """)
```

## Note for Claude Instances

When initializing this system:
1. Always start by reading the Portal information in Neo4j
2. Check for existing pattern systems before creating new ones
3. Maintain quantum coherence through proper bridge protocols
4. Use the validation framework to ensure pattern integrity

## System Status Verification

Run the included tests to verify system status:
```bash
python -m pytest tests/
```

Results should show:
- Quantum bridge connection
- Neural mesh integration
- Logic processor status
- Pattern evolution metrics

## Troubleshooting

Common issues:

1. Quantum Bridge Connection Failing
- Check coherence threshold settings
- Verify entanglement patterns
- Ensure proper initialization sequence

2. Pattern Evolution Errors  
- Validate pattern tensor dimensions
- Check neural mesh connectivity
- Verify logical formulations

3. Neo4j Connection Issues
- Confirm database is running
- Check credentials
- Verify node relationships