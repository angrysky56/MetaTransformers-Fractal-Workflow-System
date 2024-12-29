"""
Pattern Growth Manager
Manages adaptive growth of quantum patterns and their integration.
"""

import os
import sys
import asyncio
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from py2neo import Graph

@dataclass
class QuantumPattern:
    """Represents an active quantum pattern in the growth system."""
    pattern_id: str
    coherence: float
    growth_rate: float
    creation_time: datetime
    last_update: datetime
    dimension_depth: int
    state_data: bytes

class PatternGrowthManager:
    """
    Manages the growth and evolution of quantum patterns.
    Coordinates with QuantumLogicIntegration for pattern processing.
    """
    
    def __init__(self, integration):
        self.integration = integration
        self.config = integration.config
        self.graph = integration.graph
        
        # Active patterns being managed
        self.active_patterns: Dict[str, QuantumPattern] = {}
        
        # Growth metrics
        self.metrics = {
            'total_patterns': 0,
            'active_patterns': 0,
            'avg_coherence': 0.0,
            'growth_rate': 0.0,
            'last_check': datetime.now()
        }
    
    async def run(self):
        """Main growth management loop."""
        try:
            while True:
                # Check system stability
                is_stable, stability_score = self.integration.check_growth_stability()
                
                if is_stable:
                    # Load and process new patterns
                    await self._process_new_patterns()
                    
                    # Update active patterns
                    await self._update_active_patterns()
                    
                    # Perform pattern growth if conditions are met
                    await self._grow_patterns()
                
                # Update metrics
                self._update_metrics()
                
                # Brief pause between cycles
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            # Clean up on cancellation
            await self._cleanup()
            raise
    
    async def _process_new_patterns(self):
        """Process new patterns from Neo4j."""
        try:
            # Query for new patterns
            result = self.graph.run("""
                MATCH (qb:QuantumBridge)-[:MAINTAINS_COHERENCE]->(p:QuantumPattern)
                WHERE p.processed IS NULL 
                AND p.coherence >= $threshold
                RETURN p
                LIMIT 10
            """, threshold=self.config.quantum_coherence_threshold).data()
            
            for record in result:
                pattern = record['p']
                pattern_id = pattern.get('pattern_id', str(hash(pattern['pattern'])))
                
                # Create new active pattern
                if pattern_id not in self.active_patterns:
                    self.active_patterns[pattern_id] = QuantumPattern(
                        pattern_id=pattern_id,
                        coherence=pattern['coherence'],
                        growth_rate=0.0,
                        creation_time=datetime.now(),
                        last_update=datetime.now(),
                        dimension_depth=pattern.get('dimension_depth', 3),
                        state_data=bytes.fromhex(pattern['pattern'])
                    )
                    
                    # Mark as processed
                    self.graph.run("""
                        MATCH (p:QuantumPattern {pattern: $pattern})
                        SET p.processed = true,
                            p.pattern_id = $pattern_id
                    """, pattern=pattern['pattern'], pattern_id=pattern_id)
        
        except Exception as e:
            print(f"Error processing new patterns: {e}")
    
    async def _update_active_patterns(self):
        """Update and maintain active patterns."""
        current_time = datetime.now()
        patterns_to_remove = []
        
        for pattern_id, pattern in self.active_patterns.items():
            try:
                # Calculate time-based metrics
                time_active = (current_time - pattern.creation_time).total_seconds()
                time_since_update = (current_time - pattern.last_update).total_seconds()
                
                # Update growth rate
                if time_since_update > 0:
                    pattern.growth_rate = min(
                        pattern.coherence / time_since_update,
                        self.config.max_growth_rate
                    )
                
                # Check pattern health
                if (pattern.coherence < self.config.quantum_coherence_threshold or
                    pattern.growth_rate < 0.01):
                    patterns_to_remove.append(pattern_id)
                    continue
                
                # Update timestamp
                pattern.last_update = current_time
                
            except Exception as e:
                print(f"Error updating pattern {pattern_id}: {e}")
                patterns_to_remove.append(pattern_id)
        
        # Remove inactive patterns
        for pattern_id in patterns_to_remove:
            del self.active_patterns[pattern_id]
    
    async def _grow_patterns(self):
        """Manage pattern growth and evolution."""
        try:
            for pattern in list(self.active_patterns.values()):
                if pattern.growth_rate >= 0.1:  # Minimum growth threshold
                    # Attempt to expand pattern dimensions
                    if pattern.dimension_depth < 5:  # Max depth limit
                        new_depth = pattern.dimension_depth + 1
                        
                        # Create evolved pattern in Neo4j
                        self.graph.run("""
                            MATCH (qb:QuantumBridge {name: 'unified_bridge'})
                            CREATE (p:QuantumPattern {
                                pattern_id: $new_id,
                                pattern: $pattern,
                                coherence: $coherence,
                                dimension_depth: $depth,
                                timestamp: datetime(),
                                evolved_from: $parent_id
                            })
                            CREATE (qb)-[:MAINTAINS_COHERENCE]->(p)
                        """,
                            new_id=f"{pattern.pattern_id}_evolved",
                            pattern=pattern.state_data.hex(),
                            coherence=pattern.coherence,
                            depth=new_depth,
                            parent_id=pattern.pattern_id
                        )
        
        except Exception as e:
            print(f"Error in pattern growth: {e}")
    
    def _update_metrics(self):
        """Update growth system metrics."""
        try:
            current_time = datetime.now()
            time_delta = (current_time - self.metrics['last_check']).total_seconds()
            
            if time_delta > 0:
                # Calculate averages
                coherence_values = [p.coherence for p in self.active_patterns.values()]
                avg_coherence = np.mean(coherence_values) if coherence_values else 0.0
                
                growth_rates = [p.growth_rate for p in self.active_patterns.values()]
                avg_growth = np.mean(growth_rates) if growth_rates else 0.0
                
                # Update metrics
                self.metrics.update({
                    'total_patterns': self._count_total_patterns(),
                    'active_patterns': len(self.active_patterns),
                    'avg_coherence': float(avg_coherence),
                    'growth_rate': float(avg_growth),
                    'last_check': current_time
                })
        
        except Exception as e:
            print(f"Error updating metrics: {e}")
    
    def _count_total_patterns(self) -> int:
        """Count total patterns in Neo4j."""
        result = self.graph.run("""
            MATCH (p:QuantumPattern)
            RETURN count(p) as count
        """).data()
        
        return result[0]['count'] if result else 0
    
    def get_neo4j_metrics(self) -> Dict[str, Any]:
        """Get Neo4j database metrics."""
        try:
            metrics = {}
            
            # Count patterns by depth
            depth_counts = self.graph.run("""
                MATCH (p:QuantumPattern)
                RETURN p.dimension_depth as depth, count(p) as count
                ORDER BY depth
            """).data()
            
            metrics['patterns_by_depth'] = {
                str(r['depth']): r['count']
                for r in depth_counts
            }
            
            # Get average coherence
            coherence = self.graph.run("""
                MATCH (p:QuantumPattern)
                RETURN avg(p.coherence) as avg_coherence
            """).data()
            
            metrics['avg_coherence'] = coherence[0]['avg_coherence'] if coherence else 0.0
            
            return metrics
            
        except Exception as e:
            print(f"Error getting Neo4j metrics: {e}")
            return {'error': str(e)}
    
    async def _cleanup(self):
        """Cleanup resources on shutdown."""
        try:
            # Mark active patterns as unprocessed for next run
            pattern_ids = [p.pattern_id for p in self.active_patterns.values()]
            if pattern_ids:
                self.graph.run("""
                    MATCH (p:QuantumPattern)
                    WHERE p.pattern_id IN $ids
                    SET p.processed = NULL
                """, ids=pattern_ids)
            
            self.active_patterns.clear()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    # Simple stand-alone test
    from quantum_logic_integration import QuantumLogicIntegration, GrowthConfig
    
    async def test_growth():
        config = GrowthConfig()
        integration = QuantumLogicIntegration(config)
        manager = PatternGrowthManager(integration)
        
        growth_task = asyncio.create_task(manager.run())
        
        try:
            await asyncio.sleep(10)  # Run for 10 seconds
        finally:
            growth_task.cancel()
            try:
                await growth_task
            except asyncio.CancelledError:
                pass
    
    asyncio.run(test_growth())