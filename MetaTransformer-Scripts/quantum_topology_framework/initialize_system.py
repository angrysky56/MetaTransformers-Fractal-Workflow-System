import os
import sys
from typing import Dict, Any
from config_manager import QuantumConfigManager
from integration_manager import QuantumTopologyManager

class SystemInitializer:
    """
    Advanced initialization system for quantum topology framework
    """
    def __init__(self):
        self.config_manager = QuantumConfigManager()
        self.paths = self.config_manager.get_framework_paths()
        self.ensure_system_paths()
        
    def ensure_system_paths(self):
        """Ensure all required system paths exist"""
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
            
    def initialize_database_connection(self, connection_config: Dict[str, Any]) -> QuantumTopologyManager:
        """Initialize database connection with configuration"""
        return QuantumTopologyManager(
            uri=connection_config['neo4j_uri'],
            user=connection_config['user'],
            password=connection_config['password']
        )
        
    def create_default_workflow(self, manager: QuantumTopologyManager) -> str:
        """Create and register default workflow"""
        workflow_name = "default_quantum_topology_workflow"
        
        # Get configurations
        quantum_config = self.config_manager.get_config('quantum')
        topology_config = self.config_manager.get_config('topology')
        
        # Create workflow configuration
        workflow_path = self.config_manager.create_workflow_config(
            workflow_name=workflow_name,
            quantum_settings=quantum_config,
            topology_settings=topology_config
        )
        
        # Register workflow
        manager.register_measurement_workflow(
            workflow_name=workflow_name,
            workflow_config=quantum_config | topology_config
        )
        
        return workflow_path
        
    def initialize_system(self, connection_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete system initialization
        
        Args:
            connection_config: Neo4j connection configuration
            
        Returns:
            Dict containing initialization status and system information
        """
        try:
            # Initialize database connection
            manager = self.initialize_database_connection(connection_config)
            
            # Create default workflow
            workflow_path = self.create_default_workflow(manager)
            
            # Get system status
            status = manager.get_enhanced_system_status()
            
            return {
                'initialization_status': 'success',
                'system_paths': self.paths,
                'workflow_path': workflow_path,
                'system_status': status
            }
            
        except Exception as e:
            return {
                'initialization_status': 'failed',
                'error': str(e),
                'system_paths': self.paths
            }
            
def main():
    """Main initialization entry point"""
    # Load connection configuration
    config_manager = QuantumConfigManager()
    connection_config = config_manager.get_config('integration')
    
    # Add required credentials
    connection_config.update({
        'user': os.getenv('NEO4J_USER', 'neo4j'),
        'password': os.getenv('NEO4J_PASSWORD', 'password')
    })
    
    # Initialize system
    initializer = SystemInitializer()
    status = initializer.initialize_system(connection_config)
    
    # Print status
    print("\n=== Quantum Topology Framework Initialization ===")
    print(f"Status: {status['initialization_status']}")
    print("\nSystem Paths:")
    for path_name, path in status['system_paths'].items():
        print(f"  {path_name}: {path}")
    
    if status['initialization_status'] == 'success':
        print("\nWorkflow Initialized:")
        print(f"  Path: {status['workflow_path']}")
        print("\nSystem Status:")
        for key, value in status['system_status'].items():
            print(f"  {key}: {value}")
            
if __name__ == "__main__":
    main()