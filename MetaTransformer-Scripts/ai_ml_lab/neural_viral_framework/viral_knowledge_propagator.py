from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

@dataclass
class KnowledgePacket:
    """Represents a viral knowledge unit that can self-propagate"""
    id: str
    content: Dict
    propagation_rules: List[str]
    affinity_score: float
    mutation_rate: float
    generation: int

class PropagationState(Enum):
    DORMANT = "dormant"
    ACTIVE = "active"
    SPREADING = "spreading"
    INTEGRATING = "integrating"

class ViralKnowledgePropagator:
    """
    Neural-viral knowledge propagation system that mimics biological spreading patterns
    for distributed AI knowledge integration
    """
    def __init__(self, 
                 base_affinity: float = 0.75,
                 mutation_threshold: float = 0.1,
                 max_generations: int = 100):
        self.base_affinity = base_affinity
        self.mutation_threshold = mutation_threshold
        self.max_generations = max_generations
        self.knowledge_packets: Dict[str, KnowledgePacket] = {}
        self.propagation_history: List[Dict] = []
        self.setup_logging()

    def setup_logging(self):
        """Initialize logging system"""
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=log_path / "viral_propagation.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def create_knowledge_packet(self, 
                              content: Dict,
                              propagation_rules: List[str]) -> KnowledgePacket:
        """Create a new viral knowledge packet"""
        packet_id = f"kp_{len(self.knowledge_packets)}"
        packet = KnowledgePacket(
            id=packet_id,
            content=content,
            propagation_rules=propagation_rules,
            affinity_score=self.base_affinity,
            mutation_rate=0.0,
            generation=0
        )
        self.knowledge_packets[packet_id] = packet
        return packet

    def propagate_knowledge(self, 
                          packet: KnowledgePacket,
                          target_nodes: List[str]) -> List[KnowledgePacket]:
        """Propagate knowledge packet to target nodes with viral spreading mechanics"""
        propagated_packets = []
        
        for target in target_nodes:
            try:
                # Calculate propagation probability
                prop_probability = self._calculate_propagation_probability(
                    packet, target
                )
                
                if np.random.random() < prop_probability:
                    # Create mutated copy
                    new_packet = self._create_mutated_packet(
                        packet, target
                    )
                    propagated_packets.append(new_packet)
                    
                    self.logger.info(
                        f"Knowledge packet {packet.id} propagated to {target} "
                        f"with probability {prop_probability:.2f}"
                    )
                    
            except Exception as e:
                self.logger.error(
                    f"Error propagating knowledge to {target}: {str(e)}"
                )
                
        return propagated_packets

    def _calculate_propagation_probability(self,
                                        packet: KnowledgePacket,
                                        target: str) -> float:
        """Calculate probability of successful propagation"""
        # Base probability from affinity score
        prob = packet.affinity_score
        
        # Adjust for generation decay
        prob *= (1 - (packet.generation / self.max_generations))
        
        # Adjust for mutation accumulation
        prob *= (1 - packet.mutation_rate)
        
        return max(0.0, min(1.0, prob))

    def _create_mutated_packet(self,
                             parent: KnowledgePacket,
                             target: str) -> KnowledgePacket:
        """Create a mutated copy of knowledge packet"""
        # Calculate mutation rate
        new_mutation = parent.mutation_rate + np.random.normal(0, 0.1)
        new_mutation = max(0.0, min(1.0, new_mutation))
        
        # Create mutated content
        mutated_content = self._mutate_content(
            parent.content,
            new_mutation
        )
        
        return KnowledgePacket(
            id=f"{parent.id}_m{parent.generation + 1}",
            content=mutated_content,
            propagation_rules=parent.propagation_rules,
            affinity_score=parent.affinity_score * (1 - new_mutation),
            mutation_rate=new_mutation,
            generation=parent.generation + 1
        )

    def _mutate_content(self,
                       content: Dict,
                       mutation_rate: float) -> Dict:
        """Apply mutations to knowledge content"""
        if mutation_rate < self.mutation_threshold:
            return content.copy()
            
        mutated = content.copy()
        
        # Apply random mutations based on content type
        for key, value in mutated.items():
            if isinstance(value, (int, float)):
                mutated[key] = value * (1 + np.random.normal(0, mutation_rate))
            elif isinstance(value, str):
                if np.random.random() < mutation_rate:
                    mutated[key] = f"{value}_mutated"
            elif isinstance(value, list):
                if np.random.random() < mutation_rate:
                    mutated[key] = value[:int(len(value) * (1 - mutation_rate))]
                    
        return mutated

    def get_propagation_statistics(self) -> Dict:
        """Get statistics about knowledge propagation"""
        stats = {
            'total_packets': len(self.knowledge_packets),
            'active_generations': max(p.generation for p in self.knowledge_packets.values()),
            'average_affinity': np.mean([p.affinity_score for p in self.knowledge_packets.values()]),
            'average_mutation': np.mean([p.mutation_rate for p in self.knowledge_packets.values()])
        }
        
        return stats