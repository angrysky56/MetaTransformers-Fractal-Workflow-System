"""
FOLIO Integration Runner
-----------------------
Executes FOLIO dataset integration with proper path setup.
"""

import os
import sys
from pathlib import Path

# Add paths
LOGIC_LLM_PATH = Path("F:/Logic-LLM").absolute()
sys.path.append(str(LOGIC_LLM_PATH))
sys.path.append(str(LOGIC_LLM_PATH / "models"))

from folio_integrator import FOLIOIntegrator

def main():
    print("Logic-LLM Path:", LOGIC_LLM_PATH)
    print("\nVerifying paths...")
    solver_path = LOGIC_LLM_PATH / "models" / "symbolic_solvers" / "first_order_logic.py"
    print(f"Solver path exists: {solver_path.exists()}")
    
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

if __name__ == "__main__":
    main()