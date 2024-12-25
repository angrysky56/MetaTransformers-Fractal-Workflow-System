async function runMapperTests() {
    const mapper = new ConceptMapper('.');
    
    console.log('Running concept mapper tests...');
    
    try {
        // Test 1: Initialization
        console.log('\nTest 1: Initializing mapper');
        await mapper.initialize();
        
        // Test 2: Find Related Concepts
        console.log('\nTest 2: Finding concepts related to AI Architecture');
        const related = await mapper.findRelatedConcepts('ai_architecture', {
            spatialThreshold: 2.0,
            similarityThreshold: 0.6
        });
        console.log('Related concepts:', 
            related.map(r => ({
                name: r.node.properties.name,
                distance: r.distance.toFixed(3),
                similarity: r.similarity.toFixed(3)
            }))
        );
        
        // Test 3: Map Concept Path
        console.log('\nTest 3: Mapping path between concepts');
        const path = await mapper.mapConceptPath(
            'quantum_tunnels',
            'fractal_hierarchies'
        );
        console.log('Path:', {
            direct: {
                distance: path.direct.distance.toFixed(3),
                similarity: path.direct.similarity.toFixed(3)
            },
            intermediate: path.intermediate ? {
                name: path.intermediate.node.properties.name,
                totalDistance: (path.intermediate.startDistance + 
                              path.intermediate.endDistance).toFixed(3)
            } : 'none'
        });
        
        // Test 4: Space Analysis
        console.log('\nTest 4: Analyzing concept space');
        const metrics = await mapper.analyzeConceptSpace();
        console.log('Space metrics:', {
            dimensions: metrics.dimensions,
            nodeCount: metrics.nodeCount,
            density: metrics.density.toFixed(3),
            averageClustering: Object.values(metrics.clustering)
                .reduce((sum, val) => sum + val, 0) / 
                Object.values(metrics.clustering).length
        });
        
        // Test 5: Visualization Data
        console.log('\nTest 5: Getting visualization data');
        const vizData = await mapper.visualizeSpace();
        console.log('Visualization structure:', {
            nodeCount: vizData.nodes.length,
            dimensions: vizData.dimensions,
            visualizationConfig: vizData.visualization
        });
        
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Run tests
runMapperTests();