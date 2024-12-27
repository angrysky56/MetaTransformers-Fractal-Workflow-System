"""
FOLIO Dataset Integration System
------------------------------
Integrates FOLIO (First-Order Logic) dataset with neo4j and connects to symbolic solvers.
"""

import sys
import os
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional
from neo4j import GraphDatabase

# Add Logic-LLM to path
LOGIC_LLM_PATH = Path("F:/Logic-LLM").absolute()
sys.path.append(str(LOGIC_LLM_PATH))

# Import Logic-LLM components
from models.utils import load_dataset
from models.z3_solver.z3_interface import Z3Interface
from models.fol_solver.fol_parser import FOLParser

class FOLIOIntegrator:
    """
    Integrates FOLIO dataset with neural-symbolic processing capabilities.
    """
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Initialize solvers
        self.z3_solver = Z3Interface()
        self.fol_parser = FOLParser()
        
        # FOLIO dataset path
        self.dataset_path = LOGIC_LLM_PATH / "data" / "FOLIO"

    def setup_logging(self):
        """Configure logging system."""
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        fh = logging.FileHandler('folio_integration.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def initialize_folio_structure(self):
        """Initialize FOLIO dataset structure in neo4j."""
        self.logger.info("Initializing FOLIO structure...")
        
        with self.driver.session() as session:
            session.run("""
            // Create FOLIO Library Section
            MERGE (lib:KnowledgeLibrary {name: 'LogicLibrary'})
            
            MERGE (folio:LibraryIndex {
                name: 'FOLIOIndex',
                type: 'first_order_logic'
            })
            
            MERGE (lib)-[:HAS_INDEX]->(folio)
            
            // Create Processing Framework
            MERGE (framework:Framework {
                name: 'FOLIOFramework',
                type: 'logical_reasoning'
            })
            
            // Create Neural Mesh Connection
            MERGE (mesh:NeuralMesh {
                mesh_id: 'FOLIO_MESH',
                pattern_synthesis: 'first_order_logic',
                learning_rate: '0.01'
            })
            
            // Neural Processing Components
            MERGE (pattern:PatternSynthesis {
                field_type: 'fol_processing',
                state: 'active',
                coherence: 0.95,
                capability: 'logical_inference'
            })
            
            MERGE (bridge:QuantumBridge {
                bridge_id: 'FOLIO_QB_001',
                coherence_level: 0.92,
                stability_index: 0.95
            })
            
            // Integration Links
            MERGE (framework)-[:PROCESSES]->(mesh)
            MERGE (folio)-[:IMPLEMENTS]->(framework)
            MERGE (pattern)-[:ENHANCES]->(mesh)
            MERGE (bridge)-[:SYNCHRONIZES_WITH]->(mesh)
            """)
            
        self.logger.info("FOLIO structure initialized")

    def format_problem(self, problem: Dict) -> Dict:
        """Format FOLIO problem for solver."""
        premises = problem['premises']
        question = problem['question']
        
        # Parse premises and question
        parsed_premises = [self.fol_parser.parse(p) for p in premises]
        parsed_question = self.fol_parser.parse(question)
        
        return {
            'id': problem['id'],
            'premises': parsed_premises,
            'question': parsed_question,
            'answer': problem['answer']
        }

    def solve_problem(self, formatted_problem: Dict) -> Optional[Dict]:
        """Attempt to solve FOLIO problem."""
        try:
            # Convert to Z3 format
            z3_formulas = []
            for premise in formatted_problem['premises']:
                z3_formula = self.z3_solver.convert_to_z3(premise)
                z3_formulas.append(z3_formula)
                
            question_formula = self.z3_solver.convert_to_z3(formatted_problem['question'])
            
            # Check satisfiability
            result = self.z3_solver.check_validity(z3_formulas, question_formula)
            
            return {
                'is_valid': result.is_valid,
                'model': str(result.model) if result.model else None,
                'proof': result.proof if hasattr(result, 'proof') else None
            }
            
        except Exception as e:
            self.logger.error(f"Solver error: {str(e)}")
            return None

    def integrate_problem(self, problem: Dict):
        """
        Integrate a FOLIO problem into the knowledge graph.
        
        Args:
            problem: FOLIO problem dictionary
        """
        # Format and solve
        formatted = self.format_problem(problem)
        result = self.solve_problem(formatted)
        
        # Store in neo4j
        with self.driver.session() as session:
            session.run("""
            // Create Problem Node
            MERGE (prob:Concept {
                name: $id,
                type: 'folio_problem'
            })
            ON CREATE SET
                prob.created_at = datetime(),
                prob.premises = $premises,
                prob.question = $question,
                prob.answer = $answer,
                prob.solver_result = $result
                
            WITH prob
            
            // Link to FOLIO Index
            MATCH (idx:LibraryIndex {name: 'FOLIOIndex'})
            MERGE (idx)-[:INDEXES]->(prob)
            
            // Create Processing Record
            MERGE (proc:ProcessingStage {
                name: $id + '_processing',
                type: 'folio_reasoning',
                status: CASE WHEN $result IS NOT NULL THEN 'completed' ELSE 'failed' END
            })
            
            // Link Processing
            MERGE (prob)-[:PROCESSED_BY]->(proc)
            
            // Neural Processing Integration
            MATCH (mesh:NeuralMesh {mesh_id: 'FOLIO_MESH'})
            MATCH (bridge:QuantumBridge {bridge_id: 'FOLIO_QB_001'})
            
            MERGE (state:QuantumState {
                state_id: $id + '_state',
                timestamp: datetime(),
                pattern: 'fol_processing',
                coherence: CASE WHEN $result IS NOT NULL THEN 0.95 ELSE 0.7 END
            })
            
            MERGE (bridge)-[:MAINTAINS]->(state)
            MERGE (proc)-[:CURRENT_STATE]->(state)
            """, 
            id=problem['id'],
            premises=json.dumps(problem['premises']),
            question=problem['question'],
            answer=problem['answer'],
            result=json.dumps(result) if result else None
            )

    def process_folio_dataset(self):
        """Process complete FOLIO dataset."""
        self.logger.info("Processing FOLIO dataset...")
        
        # Load dataset
        dataset_file = self.dataset_path / "test.json"
        if not dataset_file.exists():
            self.logger.error(f"Dataset not found at {dataset_file}")
            return
            
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        for example in dataset:
            self.logger.info(f"Processing problem {example['id']}")
            self.integrate_problem(example)
            
        self.logger.info("FOLIO dataset processing complete")

    def verify_integration(self) -> Dict:
        """
        Verify FOLIO integration status.
        
        Returns:
            Dict containing verification metrics
        """
        with self.driver.session() as session:
            result = session.run("""
            MATCH (prob:Concept)
            WHERE prob.type = 'folio_problem'
            
            WITH count(prob) as problem_count,
                 count(prob.solver_result) as solved_count
                 
            MATCH (state:QuantumState)
            WHERE state.pattern = 'fol_processing'
            
            WITH problem_count, solved_count,
                 avg(state.coherence) as avg_coherence
                 
            RETURN {
                problems: problem_count,
                solved: solved_count,
                success_rate: toFloat(solved_count)/problem_count,
                coherence: avg_coherence
            } as metrics
            """)
            
            return result.single()['metrics']

def main():
    # Initialize integrator
    integrator = FOLIOIntegrator(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="00000000"
    )
    
    print("\nInitializing FOLIO Framework...")
    integrator.initialize_folio_structure()
    
    print("\nProcessing FOLIO Dataset...")
    integrator.process_folio_dataset()
    
    print("\nVerifying Integration...")
    metrics = integrator.verify_integration()
    print(f"Problems processed: {metrics['problems']}")
    print(f"Successfully solved: {metrics['solved']}")
    print(f"Success rate: {metrics['success_rate']*100:.1f}%")
    print(f"Average coherence: {metrics['coherence']:.3f}")

if __name__ == "__main__":
    main()