class HybridNN:
    def __init__(self):
        self.quantum_bridge = QuantumBridge(coherence_threshold=0.85)
        self.stdp_network = STDPNetwork(target_spike_rate=0.35)
        self.memory_storage = PatternStorage(entropy_norm=2.0)
