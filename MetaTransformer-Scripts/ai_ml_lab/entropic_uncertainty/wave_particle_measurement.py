"""
Quantum Wave-Particle Measurement System
--------------------------------------

Advanced implementation of quantum measurement operators for wave-particle duality analysis,
following the mathematical framework established in Spegel-Lexne et al. (2024).

Core Measurement Framework:
1. Quantum State Preparation
2. Basis Transformation
3. Probabilistic Measurement
4. Statistical Analysis

Mathematical Foundation:
- Measurement Operators: M = U†PU
- State Evolution: |ψ⟩ → U|ψ⟩
- Probability Distribution: p(i) = ⟨ψ|M†M|ψ⟩
"""

import numpy as np
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from .entropy_core import EntropicUncertainty

class WaveParticleMeasurement:
    """
    Implementation of quantum measurement operators for wave-particle duality analysis.
    
    Attributes:
        device: Computation device (GPU/CPU)
        entropy_calculator: Instance of EntropicUncertainty for calculations
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize quantum measurement system with device management.
        
        Parameters:
            device: Target computation device
            
        Implementation Note:
            Ensures consistent device usage across all quantum operations
        """
        self.device = device
        self.entropy_calculator = EntropicUncertainty(device)
        
    def _ensure_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensor compatibility for quantum operations.
        
        Parameters:
            x: Input tensor
            
        Returns:
            Device-compatible tensor with float32 precision
        """
        return x.to(self.device, dtype=torch.float32)
        
    def create_measurement_basis(self, phi: float) -> torch.Tensor:
        """
        Create quantum measurement basis for phase-dependent analysis.
        
        Parameters:
            phi: Phase angle in radians
            
        Returns:
            2x2 unitary transformation matrix
            
        Mathematical Form:
            U(φ) = [cos(φ/2)  sin(φ/2)]
                   [-sin(φ/2) cos(φ/2)]
        """
        basis = torch.tensor([
            [np.cos(phi/2), np.sin(phi/2)],
            [-np.sin(phi/2), np.cos(phi/2)]
        ], device=self.device, dtype=torch.float32)
        return basis

    def measure_state(self, 
                     state: torch.Tensor, 
                     basis: torch.Tensor) -> torch.Tensor:
        """
        Perform quantum measurement in specified basis.
        
        Parameters:
            state: Quantum state vector |ψ⟩
            basis: Measurement basis matrix U
            
        Returns:
            Measurement probability distribution
            
        Implementation:
            1. Project state onto measurement basis
            2. Calculate probability amplitudes
            3. Return normalized probabilities
        """
        state = self._ensure_tensor(state)
        basis = self._ensure_tensor(basis)
        projected = torch.matmul(basis, state)
        probabilities = torch.abs(projected)**2
        return probabilities

    def compute_interference_pattern(self,
                                   state: torch.Tensor,
                                   phi_range: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate quantum interference pattern through phase-dependent measurements.
        
        Parameters:
            state: Input quantum state
            phi_range: Range of phase angles
            
        Returns:
            Dictionary containing:
                - phases: Measurement phases
                - intensities: Interference pattern
                - visibility: Pattern visibility
                - coherence: Quantum coherence measure
        """
        state = self._ensure_tensor(state)
        phi_range = self._ensure_tensor(phi_range)
        
        intensities = []
        for phi in phi_range:
            basis = self.create_measurement_basis(phi)
            probs = self.measure_state(state, basis)
            intensities.append(probs[0].item())  # Primary detector
            
        intensities = torch.tensor(intensities, device=self.device)
        visibility = self.entropy_calculator.calculate_visibility(
            torch.max(intensities).item(),
            torch.min(intensities).item()
        )
        
        coherence = torch.mean(torch.sqrt(intensities))
        
        return {
            'phases': phi_range,
            'intensities': intensities,
            'visibility': visibility,
            'coherence': coherence.item()
        }

    def analyze_quantum_properties(self,
                                 state: torch.Tensor,
                                 basis_angles: torch.Tensor) -> Dict[str, float]:
        """
        Comprehensive analysis of quantum state properties.
        
        Parameters:
            state: Quantum state vector
            basis_angles: Set of measurement angles
            
        Returns:
            Dictionary of quantum measurements:
                - visibility: Wave-like behavior
                - distinguishability: Particle-like behavior
                - coherence: Quantum coherence
                - purity: State purity measure
                - complementarity: V²+D² validation
        """
        state = self._ensure_tensor(state)
        basis_angles = self._ensure_tensor(basis_angles)
        
        # Interference pattern analysis
        interference_data = self.compute_interference_pattern(state, basis_angles)
        visibility = interference_data['visibility']
        
        # Path distinguishability
        computational_basis = torch.eye(2, device=self.device)
        path_probs = self.measure_state(state, computational_basis)
        distinguishability = self.entropy_calculator.calculate_distinguishability(
            path_probs[0].item(),
            path_probs[1].item()
        )
        
        # Calculate state purity
        purity = torch.sum(path_probs**2).item()
        
        # Verify complementarity relation
        complementarity_value = visibility**2 + distinguishability**2
        
        return {
            'visibility': visibility,
            'distinguishability': distinguishability,
            'coherence': interference_data['coherence'],
            'purity': purity,
            'complementarity': complementarity_value
        }