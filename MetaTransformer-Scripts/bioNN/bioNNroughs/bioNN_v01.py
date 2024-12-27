import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

class Node:
    def __init__(self, id, threshold=0.5):
        self.id = id
        self.state = 0  # 0 = inactive, 1 = active
        self.threshold = threshold
        self.input_signal = 0
        self.refractory_timer = 0  # Refractory period counter

    def update_state(self):
        if self.refractory_timer > 0:  # If refractory, stay inactive
            self.state = 0
            self.refractory_timer -= 1
        elif self.input_signal > self.threshold:  # Activate if signal exceeds threshold
            self.state = 1
            self.refractory_timer = 3  # Set refractory period
        else:
            self.state = 0
        self.input_signal = 0  # Reset signal after processing

class DynamicNetwork:
    def __init__(self, num_nodes):
        self.graph = nx.Graph()
        for i in range(num_nodes):
            self.graph.add_node(i, node=Node(i, threshold=random.uniform(0.4, 0.6)))
        for _ in range(num_nodes * 2):  # Add random initial connections
            self.graph.add_edge(random.randint(0, num_nodes-1), random.randint(0, num_nodes-1), weight=random.uniform(0.1, 1.0))

    def propagate_signal(self):
        for edge in self.graph.edges(data=True):
            source, target = edge[0], edge[1]
            signal = self.graph.nodes[source]['node'].state * edge[2]['weight']
            self.graph.nodes[target]['node'].input_signal += signal
        
        for node_id in self.graph.nodes:
            self.graph.nodes[node_id]['node'].update_state()

    def adjust_weights(self):
        for edge in self.graph.edges(data=True):
            source, target = edge[0], edge[1]
            if self.graph.nodes[source]['node'].state == 1 and self.graph.nodes[target]['node'].state == 1:
                edge[2]['weight'] = min(edge[2].get('weight', 1.0) + 0.01, 1.0)  # Cap at 1.0
            else:
                edge[2]['weight'] = max(edge[2].get('weight', 1.0) - 0.005, 0.1)  # Minimum weight of 0.1

    def visualize(self, step):
        pos = nx.spring_layout(self.graph)
        node_colors = ['red' if self.graph.nodes[n]['node'].state == 1 else 'blue' for n in self.graph.nodes]
        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors, edge_color='black', width=1.0)
        plt.title(f"Step {step}")
        plt.show()

# Initialize and run the network
net = DynamicNetwork(num_nodes=10)

for step in range(10):
    net.propagate_signal()
    net.adjust_weights()
    net.visualize(step)
