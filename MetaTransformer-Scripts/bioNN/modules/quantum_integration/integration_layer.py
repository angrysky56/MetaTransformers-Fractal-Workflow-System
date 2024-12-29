"""
Integration layer between unified quantum bridge and Neo4j graph database.
Handles state tracking, metrics, and logic processor coordination.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import torch
import uuid

from .unified_quantum_bridge import UnifiedQuantumBridge, UnifiedQuantumConfig
from .neo4j_integration import (
    initialize_quantum_integration,
    create_processing_session,
    register_quantum_state,
    update_quantum_metrics,
    link_quantum_bio_states,
    track_state_evolution,
    link_to_logic_processor
)

class IntegratedQuantumBridge:
    """
    Integration layer that connects the unified quantum bridge with Neo4j.
    Handles state persistence, metric tracking, and logic processor coordination.
    """
    
    def __init__(self, config: UnifiedQuantumConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize bridge
        self.bridge = UnifiedQuantumBridge(config).to(self.device)
        
        # Initialize Neo4j integration
        if not initialize_quantum_integration():
            raise RuntimeError("Failed to initialize Neo4j integration")
            
        # Create processing session
        self.session_config = {
            'bio_dim': config.bio_dim,
            'quantum_dim': config.quantum_dim,
            'threshold': config.coherence_threshold,
            'device': str(self.device)
        }
        
        if not create_processing_session(self.session_config):
            raise RuntimeError("Failed to create processing session")
            
        self.current_state_id = None
        self.previous_state_id = None
    
    def process_bio_state(self, bio_state: torch.Tensor, 
                         state_id: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Process biological state through quantum bridge with Neo4j integration.
        """
        try:
            # Generate state ID if not provided
            self.previous_state_id = self.current_state_id
            self.current_state_id = state_id or f"quantum_{uuid.uuid4().hex[:8]}"
            
            # Process through bridge
            quantum_state, metrics = self.bridge.bio_to_quantum(bio_state)
            
            # Register quantum state in Neo4j
            if register_quantum_state(self.current_state_id, metrics):
                # Track state evolution if we have a previous state
                if self.previous_state_id:
                    track_state_evolution(
                        self.current_state_id,
                        self.previous_state_id,
                        metrics
                    )
                
                # Update processing metrics
                update_quantum_metrics(self.bridge.get_state_metrics())
                
                # Link to logic processor if coherence is high enough
                if metrics.get('coherence', 0.0) >= self.config.coherence_threshold:
                    link_to_logic_processor(self.current_state_id)
            
            return quantum_state, metrics
            
        except Exception as e:
            logging.error(f"Failed to process bio state: {str(e)}")
            raise
    
    def get_quantum_state(self) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """
        Get current quantum state with metrics.
        """
        quantum_state, coherence = self.bridge.get_current_state()
        metrics = self.bridge.get_state_metrics()
        return quantum_state, metrics
    
    def bio_to_quantum_with_tracking(self, bio_state: torch.Tensor, 
                                   bio_id: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Convert biological to quantum state with full state tracking.
        """
        quantum_state, metrics = self.process_bio_state(bio_state)
        
        if self.current_state_id and bio_id:
            link_quantum_bio_states(self.current_state_id, bio_id, metrics)
            
        return quantum_state, metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive processing metrics.
        """
        bridge_metrics = self.bridge.get_state_metrics()
        
        metrics = {
            'quantum_metrics': bridge_metrics,
            'current_state_id': self.current_state_id,
            'previous_state_id': self.previous_state_id
        }
        
        return metrics