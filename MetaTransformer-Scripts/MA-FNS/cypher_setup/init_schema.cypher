// MA-FNS Neo4j Schema Initialization

// Constraints
CREATE CONSTRAINT IF NOT EXISTS ON (n:NeuralMesh) ASSERT n.mesh_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS ON (n:QuantumBridge) ASSERT n.bridge_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS ON (n:RLAgent) ASSERT n.agent_id IS UNIQUE;

// Base System Structure
MERGE (mas:MultiAgentSystem {
    name: 'QuantumMARLController',
    num_agents: 4,
    sync_rate: 0.1,
    created: datetime()
})

// Create initial agents
UNWIND range(0, 3) as id
MERGE (agent:RLAgent {
    agent_id: 'AGENT-' + toString(id),
    local_coherence: 0.8,
    exploration_rate: 0.2,
    learning_rate: 0.01
});

// Pattern System
MERGE (ps:PatternSystem {
    name: 'FractalPatternController',
    pattern_depth: 3,
    coherence_threshold: 0.7
});

// Quantum Components
MERGE (qb:QuantumBridge {
    bridge_id: 'QB-MAIN',
    coherence_level: 0.95,
    entanglement_pattern: 'MULTI_SYSTEM'
});

MERGE (nm:NeuralMesh {
    mesh_id: 'NM-MAIN',
    pattern_synthesis: 'FRACTAL_EVOLUTION',
    learning_rate: '0.001',
    substrate: 'QUANTUM_FIELD'
});

// Link components
MATCH (qb:QuantumBridge {bridge_id: 'QB-MAIN'})
MATCH (nm:NeuralMesh {mesh_id: 'NM-MAIN'})
MERGE (qb)-[:SYNCHRONIZES_WITH]->(nm);

// Create indexes for performance
CREATE INDEX IF NOT EXISTS FOR (n:NeuralMesh) ON (n.pattern_synthesis);
CREATE INDEX IF NOT EXISTS FOR (n:QuantumBridge) ON (n.coherence_level);
CREATE INDEX IF NOT EXISTS FOR (n:RLAgent) ON (n.local_coherence);