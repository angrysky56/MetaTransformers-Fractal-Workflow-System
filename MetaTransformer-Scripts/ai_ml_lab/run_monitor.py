"""
Quantum Measurement Monitor - Runtime Framework
--------------------------------------------
Advanced monitoring system for entropic uncertainty measurements
and fractal neural mesh evolution within quantum frameworks.
"""
import os
from quantum_monitor import QuantumMonitor

def main():
    # Core system parameters
    SYSTEM_PARAMS = {
        'uri': "bolt://localhost:7687",
        'user': "neo4j",
        'password': "00000000", # Replace with users Neo4j password
        'monitor_interval': 30  # seconds
    }

    print("""
+----------------------------------------+
|      Quantum Measurement Monitor        |
|----------------------------------------|
| Monitoring:                            |
|  - Quantum State Coherence             |
|  - Neural Mesh Evolution               |
|  - Pattern Synthesis Adaptation        |
|  - Temporal Stability                  |
+----------------------------------------+
    """)

    try:
        # Initialize monitoring framework
        monitor = QuantumMonitor(
            uri=SYSTEM_PARAMS['uri'],
            user=SYSTEM_PARAMS['user'],
            password=SYSTEM_PARAMS['password']
        )

        print("\nInitializing quantum measurement framework...")
        print("Press Ctrl+C to terminate monitoring cycle")
        print("-" * 50)

        # Activate monitoring cycle
        monitor.monitor_cycle(interval=SYSTEM_PARAMS['monitor_interval'])

    except KeyboardInterrupt:
        print("\nMonitoring cycle terminated by user")
    except Exception as e:
        print(f"\nSystem Error: {str(e)}")
        print("Verify Neo4j connection parameters and system state")

if __name__ == "__main__":
    main()
