"""
Integration test script for Scale-Agnostic ML Module
"""
import sys
import torch
import yaml
from pathlib import Path
from py2neo import Graph
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment():
    """Verify environment configuration"""
    logger.info("Testing environment setup...")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")
    if cuda_available:
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # Import core modules
    try:
        from scale_agnostic_unconditional_generation.src.models.trainer import ScaleAgnosticTrainer
        from scale_agnostic_unconditional_generation.src.models.diffusion.sampler import ScaleAgnosticSampler
        logger.info("Core modules imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import core modules: {e}")
        return False

    return True

def test_database_connection():
    """Verify Neo4j database connection and structure"""
    logger.info("Testing database connection...")

    try:
        # Load configuration
        with open('lab_config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Connect to Neo4j
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

        # Verify ML Lab structure
        query = """
        MATCH (lab:MLLab)-[:CONTAINS]->(module:MLModule)
        WHERE module.name = 'scale_agnostic_generation'
        RETURN lab, module
        """
        result = graph.run(query).data()

        if result:
            logger.info("Database structure verified")
            return True
        else:
            logger.error("Failed to verify database structure")
            return False

    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

def test_integration():
    """Run complete integration test suite"""
    tests = [
        ("Environment", test_environment),
        ("Database", test_database_connection)
    ]

    results = {}
    for name, test_fn in tests:
        logger.info(f"\nRunning {name} test...")
        try:
            result = test_fn()
            results[name] = result
            status = "PASSED" if result else "FAILED"
            logger.info(f"{name} test: {status}")
        except Exception as e:
            results[name] = False
            logger.error(f"{name} test failed with error: {e}")

    return all(results.values())

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
