"""
Wave-Particle Duality Measurement System Implementation
"""
import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MeasurementConfig:
    phase_steps: int = 100
    phase_range: Tuple[float, float] = (0, 2*np.pi)
    measurement_time: float = 0.8
    integration_window: float = 3e-9

class InterferometricSystem:
    def __init__(self, config: MeasurementConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def create_sagnac_interferometer(self, phi_s: float) -> torch.Tensor:
        bs2 = torch.tensor([
            [1j, -1],
            [-1, 1j]
        ], device=self.device, dtype=torch.complex64) / np.sqrt(2)
        
        phase = torch.tensor([
            [1, 0],
            [0, np.exp(1j*phi_s)]
        ], device=self.device, dtype=torch.complex64)
        
        transfer = torch.matmul(bs2, torch.matmul(phase, bs2))
        return transfer

    def _apply_poisson_noise(self, probabilities: torch.Tensor) -> torch.Tensor:
        mean_counts = probabilities * self.config.measurement_time * 1000
        return torch.poisson(mean_counts)

    def measure_interference_pattern(self, state: torch.Tensor, phi_s: float, phi_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if len(state.shape) > 1:
            state = state[0]
        state = state.to(device=self.device, dtype=torch.complex64)
        interferometer = self.create_sagnac_interferometer(phi_s)
        
        counts_d1 = []
        counts_d2 = []
        
        # Pre-compute phase factors to avoid repeated tensor construction
        phase_factors = torch.exp(1j * phi_x.to(device=self.device))
        
        for phase_factor in phase_factors:
            # More efficient tensor operations
            modulated_state = torch.stack([
                state[0],
                state[1] * phase_factor
            ]).to(dtype=torch.complex64, device=self.device)
            
            output_state = torch.matmul(interferometer, modulated_state)
            probs = torch.abs(output_state)**2
            
            counts = self._apply_poisson_noise(probs)
            counts_d1.append(counts[0].item())
            counts_d2.append(counts[1].item())
        
        return {
            'D1': torch.tensor(counts_d1, device=self.device),
            'D2': torch.tensor(counts_d2, device=self.device)
        }

    @staticmethod
    def analyze_visibility(counts: Dict[str, torch.Tensor]) -> float:
        total_counts = counts['D1'] + counts['D2']
        normalized_d1 = counts['D1'] / (total_counts + 1e-10)
        
        max_prob = torch.max(normalized_d1)
        min_prob = torch.min(normalized_d1)
        
        visibility = (max_prob - min_prob) / (max_prob + min_prob + 1e-10)
        return visibility.item()

    def measure_distinguishability(self, state: torch.Tensor, phi_s: float) -> float:
        if len(state.shape) > 1:
            state = state[0]
        state = state.to(device=self.device, dtype=torch.complex64)
        
        # More efficient tensor construction for blocked states
        blocked_0 = torch.zeros(2, dtype=torch.complex64, device=self.device)
        blocked_1 = torch.zeros(2, dtype=torch.complex64, device=self.device)
        blocked_0[1] = state[1]
        blocked_1[0] = state[0]
        
        phase = torch.tensor([0.0], device=self.device)
        counts_0 = self.measure_interference_pattern(blocked_0, phi_s, phase)
        counts_1 = self.measure_interference_pattern(blocked_1, phi_s, phase)
        
        p1 = torch.sum(counts_0['D1']) / (torch.sum(counts_0['D1']) + torch.sum(counts_0['D2']) + 1e-10)
        p2 = torch.sum(counts_1['D1']) / (torch.sum(counts_1['D1']) + torch.sum(counts_1['D2']) + 1e-10)
        
        return abs(p1 - p2).item()