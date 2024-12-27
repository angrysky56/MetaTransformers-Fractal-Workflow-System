"""
Network Evolution Manager
Coordinates state updates and growth for the quantum neural network
"""

from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

from .quantum_state_updater import QuantumStateUpdater
from .quantum_growth_controller import QuantumGrowthController

class NetworkEvolutionManager:
    def __init__(self, 
                 uri: str = "neo4j://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password"):
        self.state_updater = QuantumStateUpdater(uri, user, password)
        self.growth_controller = QuantumGrowthController(uri, user, password)
        self.setup_logging()
        
    def setup_logging(self):
        self.logger = logging.getLogger('network_evolution')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        handler = logging.FileHandler('network_evolution.log')
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler
        self.logger.addHandler(handler)
        
    def evolve_network(self, 
                      neuron_states: Dict[str, Dict],
                      synaptic_weights: List[Tuple],
                      entanglement_data: List[Tuple[str, str, float]]):
        """
        Evolve the network based on STDP results and quantum measurements
        
        Args:
            neuron_states: Dict mapping neuron IDs to their current states
            synaptic_weights: List of (pre_id, post_id, weight_real, weight_imag)
            entanglement_data: List of (neuron1_id, neuron2_id, entanglement_strength)
        """
        try:
            # 1. Update current states
            self.logger.info("Updating neuron states...")
            self.state_updater.update_neuron_states(neuron_states)
            
            # 2. Update synaptic weights
            self.logger.info("Updating synaptic weights...")
            self.state_updater.update_synaptic_weights(synaptic_weights)
            
            # 3. Record entanglement patterns
            self.logger.info("Recording entanglement patterns...")
            entangled_pairs = [(n1, n2) for n1, n2, _ in entanglement_data]
            avg_strength = sum(s for _, _, s in entanglement_data) / len(entanglement_data)
            self.state_updater.record_entanglement_pattern(entangled_pairs, avg_strength)
            
            # 4. Check if growth is needed
            if self.growth_controller.check_growth_needed():
                self.logger.info("Network growth triggered...")
                new_neurons = self.growth_controller.grow_network()
                if new_neurons:
                    self.logger.info(f"Created {len(new_neurons)} new neurons: {new_neurons}")
                    self.growth_controller.adjust_quantum_fields(new_neurons)
            
            # 5. Get network metrics
            growth_metrics = self.growth_controller.get_growth_metrics()
            topology = self.growth_controller.analyze_network_topology()
            
            self.logger.info("Network evolution complete.")
            self.logger.info(f"Current metrics: {growth_metrics}")
            self.logger.info(f"Topology: {topology}")
            
            return {
                'growth_metrics': growth_metrics,
                'topology': topology,
                'new_neurons': new_neurons if 'new_neurons' in locals() else []
            }
            
        except Exception as e:
            self.logger.error(f"Error during network evolution: {str(e)}")
            raise
        
    def get_network_state(self) -> Dict:
        """Get current state of the entire network"""
        try:
            network_state = self.state_updater.get_network_state()
            growth_metrics = self.growth_controller.get_growth_metrics()
            topology = self.growth_controller.analyze_network_topology()
            
            return {
                'network_state': network_state,
                'growth_metrics': growth_metrics,
                'topology': topology,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting network state: {str(e)}")
            raise
            
    def close(self):
        """Clean up resources"""
        self.state_updater.close()
        self.growth_controller.close()