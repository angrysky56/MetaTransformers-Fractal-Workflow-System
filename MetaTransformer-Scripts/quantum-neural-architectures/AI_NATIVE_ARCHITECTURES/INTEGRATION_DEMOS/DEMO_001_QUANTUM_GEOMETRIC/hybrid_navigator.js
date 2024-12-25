class HybridNavigator {
    constructor(configPath) {
        this.configPath = configPath;
        this.config = null;
        this.quantum = null;
        this.geometric = null;
        this.currentState = {
            mode: 'surface', // 'surface' or 'tunnel'
            location: null,
            depth: 0
        };
    }

    async initialize() {
        const fs = window.fs;
        
        // Load configuration
        const configContent = await fs.readFile(
            `${this.configPath}/integration_config.json`,
            { encoding: 'utf8' }
        );
        this.config = JSON.parse(configContent);

        // Initialize quantum tunnel component
        this.quantum = new QuantumTunnel(this.config.components.quantum_tunnel);
        await this.quantum.initialize();

        // Initialize geometric space component
        this.geometric = new GeometricSpace(this.config.components.geometric_space);
        await this.geometric.initialize();

        // Set initial state
        this.currentState.location = this.config.components.geometric_space.surface_point;
    }

    async exploreFromSurface(conceptId, options = {}) {
        const {
            maxDepth = 3,
            relationshipThreshold = 0.6,
            returnToSurface = true
        } = options;

        // Start in geometric space
        const surfaceLocation = await this.geometric.findConceptLocation(conceptId);
        if (!surfaceLocation) {
            throw new Error('Concept not found in geometric space');
        }

        this.currentState.location = surfaceLocation;
        this.currentState.mode = 'surface';

        // Check if deep exploration is needed
        const needsDeepDive = await this.evaluateDepthNeed(conceptId);
        if (needsDeepDive) {
            // Transition to quantum tunnel
            await this.enterTunnel(conceptId, maxDepth);
            
            // Explore and gather insights
            const explorationResults = await this.quantum.explore({
                startDepth: this.currentState.depth,
                maxDepth,
                relationshipThreshold
            });

            // Process and map new relationships
            const newRelationships = await this.processQuantumInsights(explorationResults);
            
            // Update geometric space
            await this.geometric.addRelationships(newRelationships);

            if (returnToSurface) {
                await this.returnToGeometricSpace();
            }
        }

        return {
            concept: conceptId,
            location: this.currentState.location,
            mode: this.currentState.mode,
            depth: this.currentState.depth,
            explorationComplete: true
        };
    }

    async enterTunnel(conceptId, maxDepth) {
        // Map geometric coordinates to tunnel entry point
        const entryPoint = this.mapToTunnelCoordinates(this.currentState.location);
        
        // Enter tunnel
        await this.quantum.enterAt(entryPoint, {
            concept: conceptId,
            maxDepth
        });

        this.currentState.mode = 'tunnel';
        this.currentState.depth = 1;
    }

    async returnToGeometricSpace() {
        if (this.currentState.mode !== 'tunnel') {
            return;
        }

        // Get current tunnel insights
        const tunnelState = await this.quantum.getCurrentState();
        
        // Map back to geometric coordinates
        const geometricLocation = this.mapToGeometricCoordinates(tunnelState);
        
        // Update state
        this.currentState.mode = 'surface';
        this.currentState.location = geometricLocation;
        this.currentState.depth = 0;
    }

    mapToTunnelCoordinates(geometricLocation) {
        // Find nearest connection point
        const connectionPoint = this.config.connection_points
            .sort((a, b) => {
                const distA = this.calculateDistance(geometricLocation, a.geometric_coords);
                const distB = this.calculateDistance(geometricLocation, b.geometric_coords);
                return distA - distB;
            })[0];

        return {
            entry: connectionPoint.tunnel_depth,
            coordinates: this.transformCoordinates(
                geometricLocation,
                'geometric_to_tunnel'
            )
        };
    }

    mapToGeometricCoordinates(tunnelState) {
        // Transform tunnel coordinates back to geometric space
        return this.transformCoordinates(
            tunnelState.coordinates,
            'tunnel_to_geometric'
        );
    }

    async processQuantumInsights(explorationResults) {
        const relationships = [];

        for (const result of explorationResults) {
            // Map quantum insights to geometric relationships
            const relationship = {
                source: result.concept,
                target: result.related_concept,
                type: result.relationship_type,
                strength: result.connection_strength,
                spatial_distance: this.calculateDistance(
                    result.source_coordinates,
                    result.target_coordinates
                )
            };

            // Add geometric properties
            relationship.geometric = {
                source_location: this.mapToGeometricCoordinates(result.source_coordinates),
                target_location: this.mapToGeometricCoordinates(result.target_coordinates),
                space_region: this.determineSpaceRegion(relationship.spatial_distance)
            };

            relationships.push(relationship);
        }

        return relationships;
    }

    calculateDistance(point1, point2) {
        return Math.sqrt(
            point1.reduce((sum, coord, index) => {
                return sum + Math.pow(coord - point2[index], 2);
            }, 0)
        );
    }

    transformCoordinates(coordinates, direction) {
        if (direction === 'geometric_to_tunnel') {
            return coordinates.map((coord, index) => {
                return coord * this.config.components.quantum_tunnel.depth_vector[index];
            });
        } else {
            return coordinates.map((coord, index) => {
                return coord / this.config.components.quantum_tunnel.depth_vector[index];
            });
        }
    }

    determineSpaceRegion(distance) {
        if (distance < 1.0) return 'core';
        if (distance < 2.0) return 'nearby';
        return 'distant';
    }

    async evaluateDepthNeed(conceptId) {
        // Check surface understanding
        const surfaceUnderstanding = await this.geometric.analyzeConceptConnections(conceptId);
        
        // Criteria for deep dive
        return (
            surfaceUnderstanding.connectionCount < 3 ||
            surfaceUnderstanding.averageStrength < 0.5 ||
            surfaceUnderstanding.needsExploration
        );
    }

    async getCurrentState() {
        return {
            ...this.currentState,
            quantum: this.currentState.mode === 'tunnel' ? 
                await this.quantum.getCurrentState() : null,
            geometric: this.currentState.mode === 'surface' ?
                await this.geometric.getCurrentState() : null
        };
    }
}

export default HybridNavigator;