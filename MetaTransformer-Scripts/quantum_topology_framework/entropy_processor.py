import numpy as np
from scipy.stats import entropy
from typing import List, Dict, Optional

class EntropyProcessor:
    """
    Handles quantum entropy measurements and uncertainty relations
    """
    def __init__(self, coherence_threshold: float = 0.95):
        self.coherence_threshold = coherence_threshold
        self.measurement_history: List[Dict] = []
        
    def calculate_min_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate min-entropy for a probability distribution"""
        return -np.log2(np.max(probabilities))
    
    def calculate_max_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate max-entropy for a probability distribution"""
        return 2 * np.log2(np.sum(np.sqrt(probabilities)))
    
    def entropic_uncertainty_relation(self, wave_dist: np.ndarray, particle_dist: np.ndarray) -> Dict:
        """
        Calculate entropic uncertainty relation between wave and particle measurements
        
        Args:
            wave_dist: Probability distribution for wave behavior
            particle_dist: Probability distribution for particle behavior
            
        Returns:
            Dict containing uncertainty metrics
        """
        min_entropy_particle = self.calculate_min_entropy(particle_dist)
        max_entropy_wave = self.calculate_max_entropy(wave_dist)
        
        uncertainty_sum = min_entropy_particle + max_entropy_wave
        
        result = {
            'min_entropy_particle': min_entropy_particle,
            'max_entropy_wave': max_entropy_wave,
            'uncertainty_sum': uncertainty_sum,
            'complementarity_satisfied': uncertainty_sum >= 1.0
        }
        
        self.measurement_history.append(result)
        return result
    
    def validate_coherence(self, quantum_state: np.ndarray) -> bool:
        """Validate quantum state coherence"""
        return np.abs(np.vdot(quantum_state, quantum_state) - 1.0) < 1e-10
    
    def get_measurement_statistics(self) -> Dict:
        """Analyze measurement history statistics"""
        if not self.measurement_history:
            return {}
            
        uncertainty_sums = [m['uncertainty_sum'] for m in self.measurement_history]
        return {
            'mean_uncertainty': np.mean(uncertainty_sums),
            'std_uncertainty': np.std(uncertainty_sums),
            'min_uncertainty': np.min(uncertainty_sums),
            'max_uncertainty': np.max(uncertainty_sums),
            'total_measurements': len(self.measurement_history)
        }