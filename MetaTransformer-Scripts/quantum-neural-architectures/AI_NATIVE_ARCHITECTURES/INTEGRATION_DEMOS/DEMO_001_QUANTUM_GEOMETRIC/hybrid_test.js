async function runHybridTests() {
    const navigator = new HybridNavigator('.');
    
    console.log('Running hybrid navigation tests...');
    
    try {
        // Test 1: Initialization
        console.log('\nTest 1: Initializing hybrid navigator');
        await navigator.initialize();
        const initialState = await navigator.getCurrentState();
        console.log('Initial state:', initialState);
        
        // Test 2: Surface Exploration
        console.log('\nTest 2: Exploring from surface');
        const explorationResult = await navigator.exploreFromSurface('concept_a', {
            maxDepth: 2,
            relationshipThreshold: 0.7
        });
        console.log('Exploration result:', explorationResult);
        
        // Test 3: Quantum Tunnel Transition
        console.log('\nTest 3: Testing tunnel transition');
        await navigator.enterTunnel('concept_b', 3);
        const tunnelState = await navigator.getCurrentState();
        console.log('Tunnel state:', tunnelState);
        
        // Test 4: Return to Surface
        console.log('\nTest 4: Returning to geometric space');
        await navigator.returnToGeometricSpace();
        const surfaceState = await navigator.getCurrentState();
        console.log('Surface state:', surfaceState);
        
        // Test 5: Coordinate Mapping
        console.log('\nTest 5: Testing coordinate mapping');
        const geometricCoords = [1, 1, 0];
        const tunnelCoords = navigator.mapToTunnelCoordinates(geometricCoords);
        const backToGeometric = navigator.mapToGeometricCoordinates(tunnelCoords);
        console.log('Coordinate transformation:', {
            original: geometricCoords,
            tunnel: tunnelCoords,
            transformed: backToGeometric
        });
        
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Run tests
runHybridTests();