"""
Scale-Agnostic Integration Layer for Quantum Measurements
-----------------------------------------------------
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .entropy_core import EntropicUncertainty
from .wave_particle_measurement import WaveParticleMeasurement
from .measurement_system import InterferometricSystem, MeasurementConfig

@dataclass
class IntegrationConfig:
    measurement_config: MeasurementConfig
    local_window_size: int = 64
    overlap_ratio: float = 0.5
    min_scale: int = 32
    max_scale: int = 512

class ScaleAgnosticQuantumMeasurement:
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.entropy_calculator = EntropicUncertainty()
        self.measurement_system = InterferometricSystem(config.measurement_config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def create_scale_windows(self, size: int) -> List[Tuple[int, int]]:
        windows = []
        current_scale = self.config.min_scale
        while current_scale <= min(size, self.config.max_scale):
            step_size = int(current_scale * (1 - self.config.overlap_ratio))
            start = 0
            while start + current_scale <= size:
                windows.append((start, start + current_scale))
                start += step_size
            current_scale *= 2
        return windows

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        # Reshape multi-dimensional state to standard 2-component form
        if len(state.shape) > 1:
            state = state[0]  # Take first state vector from batch
        return state.to(dtype=torch.complex64)

    def measure_local_quantum_properties(self, state: torch.Tensor, window: Tuple[int, int]) -> Dict[str, float]:
        state = self._normalize_state(state)
        phi_x = torch.linspace(0, 2*np.pi, self.config.measurement_config.phase_steps)
        
        interference_pattern = self.measurement_system.measure_interference_pattern(state, np.pi/2, phi_x)
        visibility = self.measurement_system.analyze_visibility(interference_pattern)
        distinguishability = self.measurement_system.measure_distinguishability(state, np.pi/2)

        wave_probs = interference_pattern['D1'] / (interference_pattern['D1'] + interference_pattern['D2'])
        particle_probs = torch.tensor([
            torch.sum(interference_pattern['D1']),
            torch.sum(interference_pattern['D2'])
        ])
        particle_probs = particle_probs / torch.sum(particle_probs)

        entropic_results = self.entropy_calculator.compute_entropic_bound(wave_probs, particle_probs)

        return {
            'window': window,
            'visibility': visibility,
            'distinguishability': distinguishability,
            **entropic_results
        }

    def analyze_scale_invariant_properties(self, system_state: torch.Tensor) -> Dict[str, List[Dict[str, float]]]:
        if len(system_state.shape) < 2:
            system_state = system_state.unsqueeze(-1).repeat(1, 2)
            
        windows = self.create_scale_windows(len(system_state))
        scale_results = {}

        for scale in set(w[1] - w[0] for w in windows):
            scale_measurements = []
            scale_windows = [w for w in windows if w[1] - w[0] == scale]
            for window in scale_windows:
                window_state = system_state[window[0]:window[1]]
                measurement_results = self.measure_local_quantum_properties(window_state, window)
                scale_measurements.append(measurement_results)
            scale_results[f'scale_{scale}'] = scale_measurements

        return scale_results

    def verify_scale_invariance(self, scale_results: Dict[str, List[Dict[str, float]]]) -> Dict[str, float]:
        properties = ['visibility', 'distinguishability', 'min_entropy', 'max_entropy']
        invariance_metrics = {}

        for prop in properties:
            values_across_scales = []
            for scale_measurements in scale_results.values():
                scale_values = [m[prop] for m in scale_measurements if prop in m]
                values_across_scales.extend(scale_values)
            if values_across_scales:
                values = torch.tensor(values_across_scales)
                invariance_metrics[f'{prop}_mean'] = torch.mean(values).item()
                invariance_metrics[f'{prop}_std'] = torch.std(values).item()
                invariance_metrics[f'{prop}_scale_variance'] = (torch.std(values) / torch.mean(values)).item()

        return invariance_metrics