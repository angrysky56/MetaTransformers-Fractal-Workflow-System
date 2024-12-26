"""
Entropic Uncertainty Relations Core Implementation
------------------------------------------------

A foundational framework implementing advanced quantum measurement and entropic 
uncertainty calculations following the Spegel-Lexne formalism for wave-particle 
duality relations.

Mathematical Framework:
1. Entropic Bounds: H_min(Z) + H_max(W) ≥ log₂(1/c)
2. Wave-Particle Duality: V² + D² ≤ 1
3. Quantum Complementarity: Visibility-Distinguishability Trade-off

Key Components:
- Quantum State Analysis
- Entropic Bound Calculations 
- Wave-Particle Duality Measures
"""

import numpy as np
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn

class EntropicUncertainty:
    """
    Core implementation of entropic uncertainty calculations with integrated
    quantum measurement capabilities.
    
    Attributes:
        min_entropy_history: Tracking historical min entropy values
        max_entropy_history: Tracking historical max entropy values
        device: Computation device (GPU/CPU)
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize quantum entropy calculator with device management.
        
        Parameters:
            device: Computation device (GPU/CPU)
        """
        self.min_entropy_history = []
        self.max_entropy_history = []
        self.device = device
        
    def _ensure_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensor is on correct device with float32 precision.
        Critical for consistent quantum calculations.
        """
        return x.to(self.device, dtype=torch.float32)
        
    def calculate_min_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Calculate min-entropy H_min(P) = -log₂(max_j p_j)
        
        Parameters:
            probabilities: Quantum state probability distribution
            
        Returns:
            Minimum entropy value
            
        Note:
            Critical for determining lower bounds in uncertainty relations
        """
        probabilities = self._ensure_tensor(probabilities)
        max_prob = torch.max(probabilities)
        min_entropy = -torch.log2(max_prob)
        self.min_entropy_history.append(min_entropy.item())
        return min_entropy

    def calculate_max_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Calculate max-entropy H_max(P) = 2log₂(Σ√p_j)
        
        Parameters:
            probabilities: Quantum state probability distribution
            
        Returns:
            Maximum entropy value
            
        Note:
            Essential for complementarity relations in quantum measurements
        """
        probabilities = self._ensure_tensor(probabilities)
        sqrt_probs = torch.sqrt(probabilities)
        sqrt_sum = torch.sum(sqrt_probs)
        max_entropy = 2 * torch.log2(sqrt_sum)
        self.max_entropy_history.append(max_entropy.item())
        return max_entropy

    @staticmethod
    def calculate_visibility(p_max: float, p_min: float) -> float:
        """
        Calculate interferometric visibility V = (p_max - p_min)/(p_max + p_min)
        
        Parameters:
            p_max: Maximum probability
            p_min: Minimum probability
            
        Returns:
            Visibility measure [0,1]
            
        Note:
            Key measure of wave-like behavior in quantum systems
        """
        denominator = p_max + p_min
        if denominator <= 0:
            return 0.0
        return (p_max - p_min) / denominator

    @staticmethod
    def calculate_distinguishability(p1: float, p2: float) -> float:
        """
        Calculate path distinguishability D = |p1 - p2|/(p1 + p2)
        
        Parameters:
            p1: First path probability
            p2: Second path probability
            
        Returns:
            Distinguishability measure [0,1]
            
        Note:
            Quantifies particle-like behavior in quantum systems
        """
        denominator = p1 + p2
        if denominator <= 0:
            return 0.0
        return abs(p1 - p2) / denominator

    def verify_duality_relation(self, visibility: float, distinguishability: float) -> bool:
        """
        Verify wave-particle duality relation V² + D² ≤ 1
        
        Parameters:
            visibility: Measured visibility
            distinguishability: Measured distinguishability
            
        Returns:
            Boolean indicating if duality relation holds
            
        Note:
            Fundamental test of quantum complementarity
        """
        relation_sum = visibility**2 + distinguishability**2
        return relation_sum <= 1.0 + 1e-6

    def compute_entropic_bound(self, 
                             wave_probs: torch.Tensor, 
                             particle_probs: torch.Tensor) -> Dict[str, float]:
        """
        Compute entropic uncertainty bound H_min(Z) + H_max(W) ≥ 1
        
        Parameters:
            wave_probs: Wave basis probabilities
            particle_probs: Particle basis probabilities
            
        Returns:
            Dictionary containing:
                - min_entropy: Minimum entropy value
                - max_entropy: Maximum entropy value
                - bound_sum: Sum of entropies
                - satisfies_bound: Boolean validation
                
        Note:
            Core implementation of entropic uncertainty relations
        """
        wave_probs = self._ensure_tensor(wave_probs)
        particle_probs = self._ensure_tensor(particle_probs)
        
        min_entropy = self.calculate_min_entropy(particle_probs)
        max_entropy = self.calculate_max_entropy(wave_probs)
        
        bound_sum = min_entropy + max_entropy
        satisfies_bound = bound_sum >= 1.0 - 1e-6
        
        return {
            'min_entropy': min_entropy.item(),
            'max_entropy': max_entropy.item(),
            'bound_sum': bound_sum.item(),
            'satisfies_bound': satisfies_bound
        }