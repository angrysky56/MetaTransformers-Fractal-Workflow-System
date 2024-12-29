"""
Test suite for adaptive growth system.
Tests quantum-logic integration and pattern growth.
"""

import os
import sys
import asyncio
import torch
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from bioNN.modules.quantum_integration.adaptive_growth.quantum_logic_integration import (
    QuantumLogicIntegration,
    GrowthConfig
)
from bioNN.modules.quantum_integration.adaptive_growth.pattern_growth import PatternGrowthManager
from bioNN.modules.quantum_integration.adaptive_growth.env_check import verify_environment

async def test_adaptive_growth():
    """Test the adaptive growth system."""
    print("\nInitializing Adaptive Growth Test...")

    # First verify environment
    success, status = verify_environment()
    if not success:
        print("\nEnvironment verification failed!")
        return
    
    # Initialize with test configuration
    config = GrowthConfig(
        quantum_coherence_threshold=0.85,  # More permissive for testing
        entanglement_depth=3,
        stability_threshold=0.80,
        min_pattern_confidence=0.70,
        max_growth_rate=0.3,  # Faster growth for testing
        entropy_threshold=0.3,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="00000000"
    )
    
    print("\nInitializing integration layer...")
    # Initialize integration layer
    integration = QuantumLogicIntegration(config)
    
    print("\nInitializing pattern manager...")
    # Initialize pattern manager
    growth_manager = PatternGrowthManager(integration)
    
    print("\nStarting test sequence...")
    
    # Test knowledge integration
    test_content = {
        'title': 'Test Logic Concepts',
        'url': 'test://logic',
        'sections': [
            {
                'heading': 'Basic Concepts',
                'content': [
                    'A proposition is defined as a statement that is either true or false.',
                    'An axiom is a statement that is taken to be true without proof.',
                    'A theorem is a statement that follows logically from axioms.',
                    'Modus ponens is defined as the rule of inference where if P implies Q and P is true, then Q must be true.',
                ]
            }
        ]
    }
    
    success, metrics = integration.process_new_knowledge(test_content)
    print("\nKnowledge Integration Results:")
    print(f"- Success: {success}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.3f}")
        else:
            print(f"- {key}: {value}")
    
    # Start growth manager
    growth_task = asyncio.create_task(growth_manager.run())
    
    try:
        # Let it run for a bit
        print("\nMonitoring growth...")
        for i in range(5):
            await asyncio.sleep(2)
            
            # Get current metrics
            metrics = integration.get_growth_metrics()
            print(f"\nGrowth Cycle {i+1}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"- {key}: {value:.3f}")
                else:
                    print(f"- {key}: {value}")
            
            # Check active patterns
            print(f"\nActive Patterns: {len(growth_manager.active_patterns)}")
            for pattern_id, pattern in growth_manager.active_patterns.items():
                print(f"\nPattern {pattern_id}:")
                print(f"- Coherence: {pattern.coherence:.3f}")
                print(f"- Growth Rate: {pattern.growth_rate:.3f}")
                print(f"- Age (s): {(datetime.now() - pattern.creation_time).total_seconds():.1f}")
                
            # Display Neo4j metrics
            print("\nNeo4j Status:")
            metrics = growth_manager.get_neo4j_metrics()
            for key, value in metrics.items():
                print(f"- {key}: {value}")
    
    finally:
        # Clean up
        growth_task.cancel()
        try:
            await growth_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    asyncio.run(test_adaptive_growth())