"""Configuration settings for neural-viral learning system"""

NEO4J_CONFIG = {
    'uri': 'bolt://localhost:7687',
    'user': 'neo4j',
    'password': '00000000',
    'database': 'neo4j'
}

LEARNING_CONFIG = {
    'viral_mutation_rate': 0.15,
    'neural_learning_rate': 0.01,
    'coherence_threshold': 0.85,
    'propagation_limit': 3
}