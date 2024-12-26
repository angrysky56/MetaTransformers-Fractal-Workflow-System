"""
Integration Test Suite: Scale-Agnostic EUR Framework
-------------------------------------------------

Comprehensive validation framework for quantum measurement integration,
implementing Spegel-Lexne formalism for wave-particle duality analysis.
"""

import pytest
import torch
import numpy as np
from ..measurement_system import InterferometricSystem, MeasurementConfig
from ..scale_integration import ScaleAgnosticQuantumMeasurement, IntegrationConfig

#### Test Configuration Layer ####

@pytest.fixture
def measurement_config():
    """Standard measurement configuration with optimized parameters."""
    return MeasurementConfig(
        phase_steps=50,
        phase_range=(0, 2*np.pi),
        measurement_time=0.1
    )

@pytest.fixture
def integration_config(measurement_config):
    """Scale-agnostic integration configuration."""
    return IntegrationConfig(
        measurement_config=measurement_config,
        local_window_size=32,
        overlap_ratio=0.5,
        min_scale=16,
        max_scale=64
    )

@pytest.fixture
def interferometer(measurement_config):
    """Configured interferometric system."""
    return InterferometricSystem(measurement_config)

@pytest.fixture
def scale_measurement(integration_config):
    """Initialized scale-agnostic measurement system."""
    return ScaleAgnosticQuantumMeasurement(integration_config)

#### Core Integration Tests ####

def test_interferometric_measurement(interferometer):
    """Validate core interferometric measurement capabilities."""
    # Prepare quantum superposition state
    state = torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)], dtype=torch.complex64)
    phi_s = np.pi/2  # Balanced beam splitter configuration
    phi_x = torch.linspace(0, 2*np.pi, 10)
    
    # Perform measurement
    results = interferometer.measure_interference_pattern(state, phi_s, phi_x)
    
    # Structural validation
    assert 'D1' in results and 'D2' in results
    assert len(results['D1']) == len(phi_x)
    
    # Data analysis
    pattern_d1 = results['D1'].cpu().numpy()
    pattern_d2 = results['D2'].cpu().numpy()
    total_counts = pattern_d1 + pattern_d2
    
    # Physical constraints validation
    assert np.all(pattern_d1 >= 0) and np.all(pattern_d2 >= 0)
    assert np.any(total_counts > 0)  # Verify detection events
    
    # Statistical analysis
    mean_counts = np.mean(total_counts)
    std_counts = np.std(total_counts)
    cv = std_counts/mean_counts if mean_counts > 0 else float('inf')
    assert cv < 0.5  # Coefficient of variation constraint

def test_scale_invariant_analysis(scale_measurement):
    """Verify scale-agnostic quantum property analysis."""
    # Generate test quantum state
    size = 32  # Optimized test size
    system_state = torch.zeros(size, dtype=torch.complex64)
    system_state.fill_(1/np.sqrt(2))  # Equal superposition
    
    # Perform multi-scale analysis
    results = scale_measurement.analyze_scale_invariant_properties(system_state)
    
    # Structural validation
    assert isinstance(results, dict)
    assert len(results) > 0
    
    # Physical properties validation
    for scale_results in results.values():
        assert isinstance(scale_results, list)
        for measurement in scale_results:
            # Completeness check
            required_keys = ['visibility', 'distinguishability', 'window']
            assert all(key in measurement for key in required_keys)
            
            # Physical bounds
            assert 0 <= measurement['visibility'] <= 1
            assert 0 <= measurement['distinguishability'] <= 1
            
            # Wave-particle duality relation
            duality_sum = measurement['visibility']**2 + measurement['distinguishability']**2
            assert duality_sum <= 1.1  # Include numerical tolerance

def test_scale_invariance_verification(scale_measurement):
    """Validate scale invariance metrics."""
    # Test dataset with known properties
    test_results = {
        'scale_32': [
            {'visibility': 0.8, 'distinguishability': 0.4, 
             'min_entropy': 0.5, 'max_entropy': 1.2},
            {'visibility': 0.75, 'distinguishability': 0.45,
             'min_entropy': 0.48, 'max_entropy': 1.18}
        ],
        'scale_64': [
            {'visibility': 0.78, 'distinguishability': 0.42,
             'min_entropy': 0.49, 'max_entropy': 1.19}
        ]
    }
    
    # Calculate invariance metrics
    metrics = scale_measurement.verify_scale_invariance(test_results)
    
    # Validate metric structure
    required_metrics = [
        'visibility_mean', 'visibility_std', 'visibility_scale_variance',
        'distinguishability_mean', 'distinguishability_std',
        'distinguishability_scale_variance'
    ]
    
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)
    
    # Physical constraints
    assert 0 <= metrics['visibility_mean'] <= 1
    assert 0 <= metrics['distinguishability_mean'] <= 1
    assert metrics['visibility_std'] >= 0
    assert metrics['distinguishability_std'] >= 0
    
    # Scale invariance criteria
    assert metrics['visibility_scale_variance'] < 0.2
    assert metrics['distinguishability_scale_variance'] < 0.2