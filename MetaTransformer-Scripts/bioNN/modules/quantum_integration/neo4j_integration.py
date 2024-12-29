"""
Neo4j integration for quantum processing system.
"""

from typing import Dict, Any, Optional, List
import logging

def read_neo4j_cypher(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """Execute a read-only Cypher query."""
    try:
        return read_query({"sql": query, "params": params or {}})
    except Exception as e:
        logging.error(f"Neo4j read failed: {str(e)}")
        return []

def write_neo4j_cypher(query: str, params: Optional[Dict[str, Any]] = None) -> bool:
    """Execute a write Cypher query."""
    try:
        write_query({"sql": query, "params": params or {}})
        return True
    except Exception as e:
        logging.error(f"Neo4j write failed: {str(e)}")
        return False

def create_quantum_processor_node():
    """Initialize or update quantum processor node."""
    query = """
    MERGE (p:LogicProcessor {name: 'LogicLLMProcessor', type: 'quantum_hybrid'})
    SET p.last_updated = datetime(),
        p.status = 'active',
        p.requires_quantum = true
    WITH p
    MERGE (qp:ProcessingSystem {name: 'QuantumProcessing'})
    SET qp.coherence_threshold = 0.92,
        qp.dimension_depth = 3,
        qp.last_updated = datetime()
    MERGE (p)-[r:USES]->(qp)
    RETURN p, qp
    """
    return write_neo4j_cypher(query)

def register_quantum_state(state_id: str, metrics: Dict[str, float]):
    """Register a quantum state in the graph."""
    query = """
    CREATE (s:QuantumState {
        state_id: $state_id,
        coherence: $coherence,
        uncertainty: $uncertainty,
        creation_time: datetime()
    })
    WITH s
    MATCH (p:ProcessingSystem {name: 'QuantumProcessing'})
    CREATE (s)-[r:PROCESSED_BY]->(p)
    SET r.processing_time = datetime()
    """
    return write_neo4j_cypher(query, {
        'state_id': state_id,
        'coherence': metrics.get('coherence', 0.0),
        'uncertainty': metrics.get('uncertainty', 1.0)
    })

def update_quantum_metrics(metrics: Dict[str, float]):
    """Update quantum processing metrics."""
    query = """
    MATCH (p:ProcessingSystem {name: 'QuantumProcessing'})
    SET p.avg_coherence = $avg_coherence,
        p.avg_uncertainty = $avg_uncertainty,
        p.stability = $stability,
        p.last_updated = datetime(),
        p.active_states = $active_states
    """
    return write_neo4j_cypher(query, {
        'avg_coherence': metrics.get('avg_coherence', 0.0),
        'avg_uncertainty': metrics.get('avg_uncertainty', 1.0),
        'stability': metrics.get('state_stability', 0.0),
        'active_states': metrics.get('measurement_count', 0)
    })

def link_quantum_bio_states(quantum_id: str, bio_id: str, metrics: Dict[str, float]):
    """Create relationship between quantum and biological states."""
    query = """
    MATCH (q:QuantumState {state_id: $quantum_id})
    MATCH (b:BiologicalState {state_id: $bio_id})
    MERGE (q)-[r:CONVERTS_TO]->(b)
    SET r.conversion_time = datetime(),
        r.reconstruction_error = $error,
        r.uncertainty = $uncertainty
    """
    return write_neo4j_cypher(query, {
        'quantum_id': quantum_id,
        'bio_id': bio_id,
        'error': metrics.get('reconstruction_error', 0.0),
        'uncertainty': metrics.get('uncertainty', 1.0)
    })

def create_processing_session(config: Dict[str, Any]):
    """Create a new processing session node."""
    query = """
    CREATE (s:ProcessingSession {
        start_time: datetime(),
        bio_dim: $bio_dim,
        quantum_dim: $quantum_dim,
        coherence_threshold: $threshold,
        device: $device
    })
    WITH s
    MATCH (p:ProcessingSystem {name: 'QuantumProcessing'})
    CREATE (s)-[r:USES]->(p)
    SET r.initialization_time = datetime()
    RETURN s
    """
    return write_neo4j_cypher(query, config)

def update_session_metrics(session_id: str, metrics: Dict[str, float]):
    """Update processing session metrics."""
    query = """
    MATCH (s:ProcessingSession)
    WHERE ID(s) = $session_id
    SET s.current_coherence = $coherence,
        s.current_uncertainty = $uncertainty,
        s.stability = $stability,
        s.last_update = datetime()
    """
    return write_neo4j_cypher(query, {
        'session_id': session_id,
        **metrics
    })

def link_to_logic_processor(quantum_state_id: str):
    """Link quantum state to logic processor for further processing."""
    query = """
    MATCH (q:QuantumState {state_id: $state_id})
    MATCH (l:LogicProcessor {name: 'LogicLLMProcessor'})
    MERGE (q)-[r:INPUTS_TO]->(l)
    SET r.input_time = datetime(),
        r.status = 'pending'
    """
    return write_neo4j_cypher(query, {'state_id': quantum_state_id})

def track_state_evolution(state_id: str, previous_state_id: str, metrics: Dict[str, float]):
    """Track quantum state evolution through processing steps."""
    query = """
    MATCH (current:QuantumState {state_id: $current_id})
    MATCH (previous:QuantumState {state_id: $previous_id})
    CREATE (previous)-[r:EVOLVES_TO]->(current)
    SET r.evolution_time = datetime(),
        r.coherence_delta = $coherence_delta,
        r.stability = $stability
    """
    return write_neo4j_cypher(query, {
        'current_id': state_id,
        'previous_id': previous_state_id,
        'coherence_delta': metrics.get('coherence_improvement', 0.0),
        'stability': metrics.get('state_stability', 0.0)
    })

def cleanup_old_states(max_age_hours: int = 24):
    """Clean up old quantum states that are no longer needed."""
    query = """
    MATCH (s:QuantumState)
    WHERE datetime() - s.creation_time > duration({hours: $max_age})
    AND NOT (s)-[:INPUTS_TO]->(:LogicProcessor)
    DETACH DELETE s
    """
    return write_neo4j_cypher(query, {'max_age': max_age_hours})

def get_processor_status() -> Dict[str, Any]:
    """Get current status of quantum processor."""
    query = """
    MATCH (p:ProcessingSystem {name: 'QuantumProcessing'})
    RETURN 
        p.avg_coherence as coherence,
        p.avg_uncertainty as uncertainty,
        p.stability as stability,
        p.active_states as active_states,
        p.last_updated as last_update
    """
    results = read_neo4j_cypher(query)
    return results[0] if results else {}

def get_available_logic_processors() -> List[Dict[str, Any]]:
    """Get list of available logic processors that can handle quantum states."""
    query = """
    MATCH (l:LogicProcessor)
    WHERE l.requires_quantum = true
    AND l.status = 'active'
    RETURN l.name as name,
           l.type as type,
           l.last_updated as last_update
    ORDER BY l.last_updated DESC
    """
    return read_neo4j_cypher(query)

def initialize_quantum_integration():
    """Initialize all required nodes and relationships for quantum integration."""
    # Create core processing nodes
    create_quantum_processor_node()
    
    # Clean up old states
    cleanup_old_states()
    
    # Verify logic processor connection
    processors = get_available_logic_processors()
    if not processors:
        logging.warning("No active logic processors found for quantum integration")
        return False
        
    logging.info(f"Quantum integration initialized with {len(processors)} available logic processors")
    return True