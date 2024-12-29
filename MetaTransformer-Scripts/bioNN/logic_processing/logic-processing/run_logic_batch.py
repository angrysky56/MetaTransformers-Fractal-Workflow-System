"""
Run Logic Learning Batch Process
"""

import os
import sys
import asyncio
from pathlib import Path
from loguru import logger
import yaml
import numpy as np

from initialize_logic_cluster import LogicClusterInitializer
from fractal_logic_processor import FractalLogicProcessor

# Setup logging
logger.add("logic_batch.log", rotation="500 MB")

class LogicBatchRunner:
    def __init__(self, batch_config: str = None):
        self.config = self._load_config(batch_config)
        self.initializer = LogicClusterInitializer()
        self.processor = None
        
    def _load_config(self, config_path: str = None):
        if not config_path:
            config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def initialize(self):
        """Initialize the processing cluster"""
        if not self.initializer.initialize_cluster():
            raise RuntimeError("Failed to initialize logic cluster")
            
        self.processor = FractalLogicProcessor(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        logger.info("Initialized batch processor")
    
    async def process_batch(self, batch_size: int = 10, iterations: int = 100):
        """Process a batch of patterns"""
        try:
            stats = {
                'processed': 0,
                'successful': 0,
                'failed': 0,
                'avg_coherence': 0.0
            }
            
            for i in range(iterations):
                logger.info(f"Starting batch iteration {i+1}/{iterations}")
                
                # Generate batch patterns
                patterns = self._generate_batch_patterns(batch_size)
                
                # Process each pattern
                coherence_values = []
                for pattern in patterns:
                    try:
                        result = await self.processor.process_pattern(pattern)
                        
                        if result['success']:
                            stats['successful'] += 1
                            coherence_values.append(result['coherence'])
                            
                            # Log pattern results
                            logger.info(f"Pattern processed successfully:")
                            logger.info(f"- Coherence: {result['coherence']:.3f}")
                            logger.info(f"- Entanglement Depth: {result['entanglement_depth']}")
                            
                        else:
                            stats['failed'] += 1
                            logger.warning(f"Pattern processing failed: {result.get('error')}")
                            
                    except Exception as e:
                        stats['failed'] += 1
                        logger.error(f"Error processing pattern: {str(e)}")
                        
                    stats['processed'] += 1
                
                # Update average coherence
                if coherence_values:
                    avg_coherence = sum(coherence_values) / len(coherence_values)
                    stats['avg_coherence'] = (
                        (stats['avg_coherence'] * (i) + avg_coherence) / (i + 1)
                    )
                
                # Log batch statistics
                logger.info(f"Batch {i+1} Statistics:")
                logger.info(f"- Processed: {stats['processed']}")
                logger.info(f"- Successful: {stats['successful']}")
                logger.info(f"- Failed: {stats['failed']}")
                logger.info(f"- Average Coherence: {stats['avg_coherence']:.3f}")
                
                # Optional: Sleep between batches
                await asyncio.sleep(1)
            
            return stats
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return None
    
    def _generate_batch_patterns(self, batch_size: int):
        """Generate patterns for batch processing"""
        patterns = []
        contexts = [
            "Patterns emerge from quantum fields through coherent interactions.",
            "Logical structures manifest in quantum-entangled states.",
            "Information flows through quantum channels maintaining coherence.",
            "Neural networks adapt through quantum state transitions.",
            "Knowledge patterns evolve through quantum superposition."
        ]
        
        for i in range(batch_size):
            # Create pattern with random initial state
            pattern = {
                "context": contexts[i % len(contexts)],
                "query": "How does this pattern maintain quantum coherence?",
                "pattern_state": {
                    "tensor": np.random.rand(3, 3) * 0.8,  # Initial state
                    "metadata": {
                        "type": "quantum_logical",
                        "iteration": i
                    }
                }
            }
            patterns.append(pattern)
            
        return patterns

async def main():
    # Get batch size and iterations from command line
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    runner = LogicBatchRunner()
    await runner.initialize()
    
    logger.info(f"Starting batch processing with size={batch_size}, iterations={iterations}")
    stats = await runner.process_batch(batch_size, iterations)
    
    if stats:
        logger.success("Batch processing completed successfully")
        logger.info(f"Final Statistics:")
        logger.info(f"- Total Processed: {stats['processed']}")
        logger.info(f"- Total Successful: {stats['successful']}")
        logger.info(f"- Total Failed: {stats['failed']}")
        logger.info(f"- Overall Average Coherence: {stats['avg_coherence']:.3f}")
    else:
        logger.error("Batch processing failed")

if __name__ == "__main__":
    asyncio.run(main())