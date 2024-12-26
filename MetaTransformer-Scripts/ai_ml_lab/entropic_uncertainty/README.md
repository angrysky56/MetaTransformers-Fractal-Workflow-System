# Entropic Uncertainty Relations & Wave-Particle Duality Framework

## Overview
This framework implements the experimental demonstration of entropic uncertainty relations (EUR) and their equivalence with wave-particle duality, based on Spegel-Lexne et al., Sci. Adv. 10, eadr2007 (2024).

### Core Framework Components
```
entropic_uncertainty/
├── entropy_core.py       # Core EUR calculations
├── measurement_system.py # Wave-particle measurements
└── scale_integration.py  # Scale-agnostic integration
```

## Installation & Setup

### Prerequisites
- Python 3.10+
- CUDA 12.0+ (for GPU acceleration)
- Active Python virtual environment

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv eur_env
source eur_env/bin/activate  # Unix
# or
.\eur_env\Scripts\activate   # Windows

# Install dependencies
pip install torch==2.1.0 
pip install torch-geometric
pip install numpy scipy matplotlib
```

## System Architecture

### 1. Quantum Measurement Layer
The framework implements three key measurement components:

#### 1.1 Entropic Calculations
- Min-entropy: H_min(P) = -log₂(max_j p_j)
- Max-entropy: H_max(P) = 2log₂(Σ√p_j)

#### 1.2 Wave-Particle Measures
- Visibility (V): (p_max - p_min)/(p_max + p_min)
- Distinguishability (D): 1/2(|p1 - p2|/(p1 + p2))

### 2. Scale-Agnostic Integration
Implements multi-scale analysis through:
- Overlapping window decomposition
- Scale-invariant property verification
- Local-to-global quantum measurements

## Usage Guide

### Basic Implementation
```python
from entropic_uncertainty.entropy_core import EntropicUncertainty
from entropic_uncertainty.measurement_system import (
    InterferometricSystem, 
    MeasurementConfig
)

# Initialize measurement system
config = MeasurementConfig(
    phase_steps=100,
    phase_range=(0, 2*np.pi),
    measurement_time=0.8
)
system = InterferometricSystem(config)

# Perform measurements
state = torch.tensor([1/np.sqrt(2), 1j/np.sqrt(2)])
phi_x = torch.linspace(0, 2*np.pi, 100)
results = system.measure_interference_pattern(state, np.pi/2, phi_x)
```

### Advanced Integration
```python
from entropic_uncertainty.scale_integration import (
    ScaleAgnosticQuantumMeasurement,
    IntegrationConfig
)

# Configure integration
integration_config = IntegrationConfig(
    measurement_config=config,
    local_window_size=64,
    overlap_ratio=0.5
)

# Initialize scale-agnostic measurement
measurement = ScaleAgnosticQuantumMeasurement(integration_config)

# Analyze multi-scale properties
scale_results = measurement.analyze_scale_invariant_properties(system_state)
invariance_metrics = measurement.verify_scale_invariance(scale_results)
```

## Testing & Validation

### Unit Tests
Run the test suite:
```bash
python -m pytest tests/
```

### Validation Metrics
The framework provides several validation checkpoints:
1. EUR bound verification: H_min(Z) + min_W H_max(W) ≥ 1
2. Wave-particle duality: V² + D² ≤ 1
3. Scale invariance measures

## Troubleshooting

### Common Issues
1. CUDA Compatibility
   ```python
   # Check CUDA availability
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   ```

2. Memory Management
   ```python
   # Monitor GPU memory
   print(f"Memory allocated: {torch.cuda.memory_allocated(0)}")
   print(f"Memory cached: {torch.cuda.memory_reserved(0)}")
   ```

### Performance Optimization
- Use batch processing for large-scale analysis
- Enable CUDA acceleration when available
- Optimize window sizes based on system memory

## Integration with ML Lab

### Database Connection
```python
from neo4j import GraphDatabase

# Connect to ML Lab database
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

# Register new measurements
def register_measurement(tx, results):
    tx.run("""
        MATCH (lab:MLLab {name: 'AI ML Laboratory'})
        CREATE (m:Measurement {
            timestamp: datetime(),
            visibility: $visibility,
            distinguishability: $distinguishability
        })
        CREATE (lab)-[:CONTAINS]->(m)
    """, results)
```

### Workflow Integration
1. Initialize measurement system
2. Perform quantum measurements
3. Store results in Neo4j database
4. Validate scale invariance
5. Generate analysis reports

## Further Development

### Planned Extensions
- Real-time measurement visualization
- Distributed computation support
- Advanced error analysis
- Integration with quantum simulation frameworks

### Contributing
1. Fork the repository
2. Create feature branch
3. Submit pull request with tests
4. Ensure documentation is updated

## References
1. Spegel-Lexne et al., Sci. Adv. 10, eadr2007 (2024)
2. PyTorch Geometric documentation
3. Neo4j Python driver documentation
