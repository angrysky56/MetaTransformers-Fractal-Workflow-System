import sys
import os

# Add framework path to system path
FRAMEWORK_PATH = "F:/MetaTransformers-Fractal-Workflow-System/MetaTransformer-Scripts"
if FRAMEWORK_PATH not in sys.path:
    sys.path.append(FRAMEWORK_PATH)

from neo4j import GraphDatabase
from typing import Dict, Any
import numpy as np
from quantum_topology_framework.entropy_processor import EntropyProcessor
from quantum_topology_framework.topology_processor import InfiniteDimensionalTopology

class QuantumTopologyManager:
    """
    Advanced integration system for quantum-topological processing
    """
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.entropy_processor = EntropyProcessor()
        self.topology_processor = InfiniteDimensionalTopology()
        self.framework_path = FRAMEWORK_PATH
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize and validate system components"""
        with self.driver.session() as session:
            # Verify core system nodes
            session.write_transaction(self._ensure_core_nodes)
            
    def _ensure_core_nodes(self, tx):
        """Ensure all required system nodes exist"""
        query = """
        MERGE (qm:QuantumMetaSystem {
            name: 'QuantumTopologyCore',
            version: '1.0.0',
            created: datetime(),
            framework_path: $framework_path
        })
        WITH qm
        MERGE (ep:EntropyProcessor {name: 'MainEntropyProcessor'})
        MERGE (tp:TopologyProcessor {name: 'MainTopologyProcessor'})
        MERGE (qm)-[:USES_PROCESSOR]->(ep)
        MERGE (qm)-[:USES_PROCESSOR]->(tp)
        """
        return tx.run(query, framework_path=self.framework_path)

    # [Previous methods remain the same...]

    def register_measurement_workflow(self, workflow_name: str, workflow_config: Dict):
        """Register a new measurement workflow"""
        with self.driver.session() as session:
            query = """
            MATCH (qm:QuantumMetaSystem {name: 'QuantumTopologyCore'})
            CREATE (w:QuantumWorkflow {
                name: $workflow_name,
                config: $config,
                created: datetime(),
                status: 'initialized'
            })
            CREATE (qm)-[:MANAGES_WORKFLOW]->(w)
            """
            session.run(query, 
                       workflow_name=workflow_name,
                       config=str(workflow_config))
    
    def get_enhanced_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status with workflow information"""
        basic_status = super().get_system_status()
        
        # Add workflow status
        with self.driver.session() as session:
            workflow_query = """
            MATCH (qm:QuantumMetaSystem)-[:MANAGES_WORKFLOW]->(w:QuantumWorkflow)
            RETURN w.name as name, w.status as status, w.created as created
            """
            workflows = list(session.run(workflow_query))
            
        return {
            **basic_status,
            'active_workflows': [dict(w) for w in workflows],
            'system_path': self.framework_path,
            'initialization_status': 'complete'
        }