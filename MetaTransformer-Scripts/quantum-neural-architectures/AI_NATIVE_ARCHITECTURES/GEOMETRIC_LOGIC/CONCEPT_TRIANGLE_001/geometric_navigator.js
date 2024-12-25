class GeometricNavigator {
    constructor(spacePath) {
        this.spacePath = spacePath;
        this.space = null;
        this.currentNode = null;
        this.nodes = new Map();
    }

    async initialize() {
        const fs = window.fs;
        
        // Load space configuration
        const spaceContent = await fs.readFile(`${this.spacePath}/geometric_space.json`, { encoding: 'utf8' });
        this.space = JSON.parse(spaceContent);

        // Load all nodes
        for (const vertex of this.space.geometry.vertices) {
            const nodeContent = await fs.readFile(`${this.spacePath}/nodes/node_${vertex.id}.json`, { encoding: 'utf8' });
            this.nodes.set(vertex.id, JSON.parse(nodeContent));
        }
    }

    async moveToNode(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node) {
            throw new Error(`Node ${nodeId} not found`);
        }
        this.currentNode = node;
        return node;
    }

    calculateDistance(coord1, coord2) {
        return Math.sqrt(
            Math.pow(coord2[0] - coord1[0], 2) +
            Math.pow(coord2[1] - coord1[1], 2) +
            Math.pow(coord2[2] - coord1[2], 2)
        );
    }

    async findNearest(coordinates) {
        let nearest = null;
        let minDistance = Infinity;

        for (const [id, node] of this.nodes) {
            const distance = this.calculateDistance(coordinates, node.coordinates);
            if (distance < minDistance) {
                minDistance = distance;
                nearest = node;
            }
        }

        return nearest;
    }

    async getRelated(relationshipType) {
        if (!this.currentNode) {
            throw new Error('No current node selected');
        }

        const related = [];
        for (const connection of this.currentNode.connections) {
            if (connection.relationship === relationshipType) {
                const targetNode = await this.findNodeById(connection.target);
                if (targetNode) {
                    related.push({
                        node: targetNode,
                        strength: connection.strength
                    });
                }
            }
        }

        return related;
    }

    async findNodeById(nodeId) {
        for (const [id, node] of this.nodes) {
            if (node.node_id === nodeId) {
                return node;
            }
        }
        return null;
    }

    async findPath(startId, endId) {
        const start = this.nodes.get(startId);
        const end = this.nodes.get(endId);

        if (!start || !end) {
            throw new Error('Invalid start or end node');
        }

        // Simple path finding for triangular space
        const directConnection = start.connections.find(c => c.target === end.node_id);
        
        if (directConnection) {
            return {
                path: [start, end],
                distance: this.calculateDistance(start.coordinates, end.coordinates),
                strength: directConnection.strength
            };
        }

        // Find intermediate node for indirect path
        const intermediate = Array.from(this.nodes.values()).find(node => {
            const hasStartConnection = start.connections.some(c => c.target === node.node_id);
            const hasEndConnection = node.connections.some(c => c.target === end.node_id);
            return hasStartConnection && hasEndConnection;
        });

        if (intermediate) {
            const startToInt = start.connections.find(c => c.target === intermediate.node_id);
            const intToEnd = intermediate.connections.find(c => c.target === end.node_id);
            
            return {
                path: [start, intermediate, end],
                distance: this.calculateDistance(start.coordinates, intermediate.coordinates) +
                         this.calculateDistance(intermediate.coordinates, end.coordinates),
                strength: Math.min(startToInt.strength, intToEnd.strength)
            };
        }

        throw new Error('No path found');
    }

    async getSpaceMetrics() {
        return {
            dimensions: this.space.dimensions,
            scale: this.space.scale,
            vertexCount: this.space.geometry.vertices.length,
            edgeCount: this.space.geometry.edges.length,
            currentNode: this.currentNode?.node_id
        };
    }
}

export default GeometricNavigator;