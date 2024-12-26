"""
Test Suite: Quantum Measurement System
------------------------------------

Tests the interferometric measurement system implementation based on 
Spegel-Lexne et al., Sci. Adv. 10, eadr2007 (2024).
"""

import pytest
import torch
import numpy as np
from ..measurement_system import InterferometricSystem, MeasurementConfig

@pytest.fixture
def measurement_config():
    """Provides standard measurement configuration for testing"""
    return MeasurementConfig(
        phase_steps=50,  # Reduced for testing efficiency
        phase_range=(0, 2*np.pi),
        measurement_time=0.1,  # Short integration for tests
        integration_window=3e-9
    )

@pytest.fixture
def interferometer(measurement_config):
    """Initializes interferometer system with test configuration"""
    return InterferometricSystem(measurement_config)

def test_interferometer_initialization(interferometer):
    """Verify proper initialization of interferometer system"""
    assert isinstance(interferometer.config.phase_steps, int)
    assert interferometer.config.phase_steps > 0
    assert isinstance(interferometer.config.measurement_time, float)
    assert interferometer.device in ['cuda', 'cpu']

def test_sagnac_interferometer_creation(interferometer):
    """Test Sagnac interferometer transfer matrix properties"""
    phi_s = np.pi/2  # 50-50 beam splitter configuration
    transfer = interferometer.create_sagnac_interferometer(phi_s)
    
    # Verify matrix properties
    assert torch.is_complex(transfer)
    assert transfer.shape == (2, 2)
    
    # Check unitarity
    identity = torch.eye(2, dtype=torch.complex64)
    product = torch.matmul(transfer, transfer.conj().T)
    assert torch.allclose(product, identity, atol=1e-6)

def test_interference_pattern_measurement(interferometer):
    """Test measurement of interference patterns"""
    # Prepare |+⟩ state
    state = torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)], dtype=torch.complex64)
    phi_s = np.pi/2
    phi_x = torch.linspace(0, 2*np.pi, 10)
    
    results = interferometer.measure_interference_pattern(state, phi_s, phi_x)
    
    # Verify output structure
    assert 'D1' in results
    assert 'D2' in results
    assert len(results['D1']) == len(phi_x)
    
    # Check physical constraints
    d1_counts = results['D1']
    d2_counts = results['D2']
    assert torch.all(d1_counts >= 0)  # Non-negative counts
    assert torch.all(d2_counts >= 0)
    
    # Verify total probability conservation
    total_counts = d1_counts + d2_counts
    mean_total = torch.mean(total_counts)
    assert torch.all(torch.abs(total_counts - mean_total) < mean_total * 0.2)

def test_visibility_analysis(interferometer):
    """Test visibility calculations from interference patterns"""
    # Create test interference pattern
    counts = {
        'D1': torch.tensor([100, 50, 100]),  # Max contrast pattern
        'D2': torch.tensor([50, 100, 50])
    }
    
    visibility = interferometer.analyze_visibility(counts)
    
    # Verify visibility bounds
    assert 0 <= visibility <= 1
    assert abs(visibility - 0.333) < 0.01  # Expected from test pattern

def test_distinguishability_measurement(interferometer):
    """Test path distinguishability measurements"""
    state = torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)], dtype=torch.complex64)
    phi_s = np.pi/2
    
    distinguishability = interferometer.measure_distinguishability(state, phi_s)
    
    # Verify distinguishability bounds
    assert 0 <= distinguishability <= 1
    
    # Test complementarity
    visibility = interferometer.analyze_visibility(
        interferometer.measure_interference_pattern(
            state, phi_s, torch.linspace(0, 2*np.pi, 20)
        )
    )
    
    # Verify wave-particle duality relation
    assert visibility**2 + distinguishability**2 <= 1.0 + 1e-6  # Allow small numerical error

def test_measurement_stability(interferometer):
    """Test measurement stability under repeated measurements"""
    state = torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)], dtype=torch.complex64)
    phi_s = np.pi/2
    phi_x = torch.linspace(0, 2*np.pi, 20)
    
    results_1 = interferometer.measure_interference_pattern(state, phi_s, phi_x)
    results_2 = interferometer.measure_interference_pattern(state, phi_s, phi_x)
    
    # Compare statistical properties
    mean_1 = torch.mean(results_1['D1'] + results_1['D2'])
    mean_2 = torch.mean(results_2['D1'] + results_2['D2'])
    
    # Allow for statistical fluctuations but verify reasonable stability
    assert abs(mean_1 - mean_2) < 0.2 * (mean_1 + mean_2)/2