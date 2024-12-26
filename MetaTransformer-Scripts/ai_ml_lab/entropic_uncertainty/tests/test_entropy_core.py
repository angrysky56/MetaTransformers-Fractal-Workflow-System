"""
Test suite for entropy core functionality
"""

import pytest
import torch
import numpy as np
from ..entropy_core import EntropicUncertainty
from ..wave_particle_measurement import WaveParticleMeasurement

@pytest.fixture
def entropy_calculator():
    return EntropicUncertainty('cpu')  # Force CPU for testing

@pytest.fixture
def measurement_system():
    return WaveParticleMeasurement('cpu')  # Force CPU for testing

def test_min_entropy_calculation(entropy_calculator):
    # Test with equal probabilities
    probs = torch.tensor([0.5, 0.5])
    min_entropy = entropy_calculator.calculate_min_entropy(probs)
    assert abs(min_entropy.cpu().item() - 1.0) < 1e-6
    
    # Test with certain outcome
    probs = torch.tensor([1.0, 0.0])
    min_entropy = entropy_calculator.calculate_min_entropy(probs)
    assert abs(min_entropy.cpu().item() - 0.0) < 1e-6

def test_visibility_calculation(entropy_calculator):
    visibility = entropy_calculator.calculate_visibility(0.8, 0.2)
    assert abs(visibility - 0.6) < 1e-6

def test_distinguishability_calculation(entropy_calculator):
    distinguishability = entropy_calculator.calculate_distinguishability(0.7, 0.3)
    assert abs(distinguishability - 0.4) < 1e-6

def test_duality_relation(entropy_calculator):
    # Test boundary case
    visibility = 1/np.sqrt(2)
    distinguishability = 1/np.sqrt(2)
    assert entropy_calculator.verify_duality_relation(visibility, distinguishability)
    
def test_measurement_basis(measurement_system):
    phi = 0.0
    basis = measurement_system.create_measurement_basis(phi)
    eye = torch.eye(2, device=basis.device)
    assert torch.allclose(basis, eye)

def test_state_measurement(measurement_system):
    # Test measurement of |+⟩ state
    state = torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)])
    basis = torch.eye(2, device=measurement_system.device)
    probs = measurement_system.measure_state(state, basis)
    expected = torch.tensor([0.5, 0.5], device=probs.device)
    assert torch.allclose(probs, expected)