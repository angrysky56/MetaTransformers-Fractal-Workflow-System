import torch
import torch.nn as nn
import networkx as nx
import numpy as np

# Define Node Class
class Node:
    def __init__(self, id, threshold=0.5):
        self.id = id
        self.state = 0  # Initial state
        self.threshold = threshold
        self.input_signal = 0
    
    def update_state(self):
        if self.input_signal > self.threshold:
            self.state = 1
        else:
            self.state = 0
        self.input_signal = 0  # Reset input signal after update

# Define Dynamic Network
class DynamicNetwork:
    def __init__(self, num_nodes):
        self.graph = nx.Graph()
        for i in range(num_nodes):
            self.graph.add_node(i, node=Node(i))
    
    def propagate_signal(self):
        for edge in self.graph.edges(data=True):
            source, target = edge[0], edge[1]
            signal = np.random.rand() * edge[2].get('weight', 1.0)
            self.graph.nodes[target]['node'].input_signal += signal
        
        for node_id in self.graph.nodes:
            self.graph.nodes[node_id]['node'].update_state()

    def adjust_weights(self):
        for edge in self.graph.edges(data=True):
            source, target = edge[0], edge[1]
            if self.graph.nodes[source]['node'].state == 1 and self.graph.nodes[target]['node'].state == 1:
                edge[2]['weight'] = edge[2].get('weight', 1.0) + 0.1  # Reinforce

# Create and Test Network
net = DynamicNetwork(num_nodes=10)
for i in range(10):  # Add random connections
    net.graph.add_edge(np.random.randint(10), np.random.randint(10), weight=np.random.rand())

# Run simulation
for _ in range(100):  # Simulate 100 timesteps
    net.propagate_signal()
    net.adjust_weights()
    print(f"States: {[net.graph.nodes[n]['node'].state for n in net.graph.nodes]}")
