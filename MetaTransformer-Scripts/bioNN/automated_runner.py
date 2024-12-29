"""
BioNN Automated Runner
Requires:
- NEO4J_PASSWORD environment variable
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
from pathlib import Path
import yaml
import logging
from datetime import datetime
import torch
from torch_geometric.data import Data

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

print("Starting bioNN Automated Runner...")

# Check environment variables first
if not os.getenv('NEO4J_PASSWORD'):
    print("\nError: NEO4J_PASSWORD environment variable must be set.")
    print("Please set it in your environment with:")
    print("  PowerShell: $env:NEO4J_PASSWORD='your_password'")
    print("  CMD: set NEO4J_PASSWORD=your_password")
    print("  Linux/Mac: export NEO4J_PASSWORD=your_password")
    sys.exit(1)

# Import from package
from bioNN.modules.stdp import QuantumSTDPLayer
from bioNN.modules.hybrid_processor import HybridBioQuantumProcessor
from ai_ml_lab.quantum_monitor import QuantumMonitor

class AutomatedSTDPRunner:
    def __init__(self):
        print("\nInitializing Runner...")
        self.setup_logging()
        self.load_config()
        self.setup_monitor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device: " + str(self.device))
        self.logger.info("Initialized AutomatedSTDPRunner")

    def setup_logging(self):
        self.logger = logging.getLogger('bionn_automated')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('bionn_automated.log')
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def load_config(self):
        config_path = project_root / 'ai_ml_lab' / 'lab_config.yaml'
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"Configuration loaded from {config_path}")
            if not self.config.get('environments', {}).get('bionn'):
                print("Warning: BioNN environment configuration not found")
        except Exception as e:
            print(f"Could not load config from {config_path}: {str(e)}")
            self.config = {}

    def setup_monitor(self):
        try:
            self.monitor = QuantumMonitor(
                uri="neo4j://localhost:7687",
                user="neo4j",
                password=os.getenv('NEO4J_PASSWORD')
            )
            print("Quantum monitor initialized")
        except Exception as e:
            print(f"Could not initialize quantum monitor: {str(e)}")
            self.monitor = None

    def create_test_data(self, num_nodes: int = 10, features: int = 16):
        x = torch.randn(num_nodes, features).to(self.device)
        edge_list = [[i, j] for i in range(num_nodes) 
                    for j in range(num_nodes) if i != j]
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
        
        return Data(x=x, edge_index=edge_index)

    def run_experiment(self, num_steps: int = 100):
        print("\nStarting automated STDP experiment...")
        
        try:
            # Initialize STDP layer on correct device
            stdp_layer = QuantumSTDPLayer(
                in_channels=16,
                out_channels=32,
                tau_plus=20.0,
                tau_minus=20.0,
                learning_rate=0.01,
                quantum_coupling=0.1
            ).to(self.device)
            
            # Create test data
            data = self.create_test_data()
            
            # Verify device placement
            print("\nSTDP Layer device: " + str(next(stdp_layer.parameters()).device))
            print("Input data device: " + str(data.x.device))
            print("Edge index device: " + str(data.edge_index.device))
            
            # Run steps
            for step in range(num_steps):
                # Process data
                spikes = stdp_layer(data.x, data.edge_index)
                entanglement = stdp_layer.quantum_entangle()
                coherence = stdp_layer.get_complex_weights().abs().mean().item()
                
                # Log metrics
                metrics = {
                    'timestamp': str(datetime.now()),
                    'step': str(step),
                    'spike_rate': str(spikes.mean().item()),
                    'entanglement': str(entanglement.item()),
                    'coherence_level': str(coherence)
                }
                
                # Log to quantum monitor if available
                if self.monitor:
                    self.monitor.log_measurement_event(metrics)
                
                # Update progress
                if step % 10 == 0:
                    print("\nTimestep " + str(step) + "/" + str(num_steps) + ":")
                    print("Average spike rate: " + "{:.4f}".format(float(metrics['spike_rate'])))
                    print("Quantum entanglement: " + "{:.4f}".format(float(metrics['entanglement'])))
                    print("Coherence level: " + "{:.4f}".format(float(metrics['coherence_level'])))
                
            print("\nExperiment completed successfully!")
            return True
            
        except Exception as e:
            print("\nError in experiment: " + str(e)})
            import traceback
            traceback.print_exc()
            return False

def main():
    print("\n=== BioNN Quantum STDP Experiment ===")
    runner = AutomatedSTDPRunner()
    runner.run_experiment()

if __name__ == "__main__":
    main()