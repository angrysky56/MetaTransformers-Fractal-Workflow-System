"""
Quantum Measurement System Monitor
--------------------------------
Active monitoring system for entropic uncertainty measurements and neural mesh evolution.
"""

import sys
import time
from datetime import datetime
from typing import Dict, Optional, List, Union
import logging
from neo4j import GraphDatabase
import numpy as np
import torch

class QuantumMonitor:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.setup_logging()
        
    def setup_logging(self):
        self.logger = logging.getLogger('quantum_monitor')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('quantum_measurements.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def format_metric_value(self, value: Union[str, float, int, None]) -> str:
        """Safely format metric values for logging."""
        if value is None:
            return 'N/A'
        try:
            if isinstance(value, str):
                # Try to convert string to float for formatting
                return f"{float(value):.3f}"
            return f"{float(value):.3f}"
        except (ValueError, TypeError):
            # If conversion fails, return as is
            return str(value)

    def log_measurement_event(self, metrics: Dict):
        """Log key measurement metrics"""
        coherence = self.format_metric_value(metrics.get('coherence_level'))
        stability = self.format_metric_value(metrics.get('stability'))
        learning_rate = self.format_metric_value(metrics.get('learning_rate'))
        spike_rate = self.format_metric_value(metrics.get('spike_rate'))
        entanglement = self.format_metric_value(metrics.get('entanglement'))
        
        log_msg = (
            "Measurement Event | "
            f"Coherence: {coherence} | "
            f"Stability: {stability} | "
            f"Learning Rate: {learning_rate}"
        )
        
        # Add optional STDP metrics if present
        if spike_rate != 'N/A' or entanglement != 'N/A':
            log_msg += f" | Spike Rate: {spike_rate} | Entanglement: {entanglement}"
            
        self.logger.info(log_msg)

    def monitor_cycle(self, interval: int = 60):
        """Run continuous monitoring cycle"""
        self.logger.info("Starting Quantum Measurement Monitor")
        
        try:
            while True:
                # Get current system state
                system_state = self.get_system_state()
                if not system_state:
                    self.logger.error("Failed to retrieve system state")
                    time.sleep(interval)
                    continue

                # Check learning progress
                learning_metrics = self.monitor_learning_progress()
                coherence_metrics = self.check_coherence_metrics()
                
                # Combine metrics
                metrics = {
                    'timestamp': datetime.now(),
                    'system_state': system_state['lab_state'],
                    'workflow_status': system_state['workflow_status'],
                    'coherence_level': coherence_metrics['coherence_level'],
                    'stability': coherence_metrics['stability'],
                    'learning_rate': learning_metrics['learning_rate']
                }
                
                # Log measurement event
                self.log_measurement_event(metrics)
                
                # Alert on issues
                if coherence_metrics['coherence_level'] < 0.8:
                    self.logger.warning("Low quantum coherence detected")
                if coherence_metrics['stability'] < 0.85:
                    self.logger.warning("System stability degrading")
                
                time.sleep(interval)

        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {str(e)}")
        finally:
            self.driver.close()

def main():
    # Initialize monitor with your Neo4j credentials
    monitor = QuantumMonitor(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="your_password"  # Configure this
    )
    
    # Start monitoring cycle
    monitor.monitor_cycle()

if __name__ == "__main__":
    main()