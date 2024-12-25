async function runNavigatorTests() {
    const navigator = new GeometricNavigator('.');
    
    console.log('Running geometric navigation tests...');
    
    try {
        // Test 1: Initialization
        console.log('\nTest 1: Initializing navigator');
        await navigator.initialize();
        const metrics = await navigator.getSpaceMetrics();
        console.log('Space metrics:', metrics);
        
        // Test 2: Node Navigation
        console.log('\nTest 2: Moving to node A');
        const nodeA = await navigator.moveToNode('A');
        console.log('Current node:', nodeA.concept_data.name);
        
        // Test 3: Finding Related Concepts
        console.log('\nTest 3: Finding related concepts');
        const related = await navigator.getRelated('contains');
        console.log('Related concepts:', related.map(r => r.node.concept_data.name));
        
        // Test 4: Path Finding
        console.log('\nTest 4: Finding path from A to C');
        const path = await navigator.findPath('A', 'C');
        console.log('Path:', path.path.map(n => n.concept_data.name));
        console.log('Path strength:', path.strength);
        
        // Test 5: Nearest Node
        console.log('\nTest 5: Finding nearest node to coordinates');
        const nearest = await navigator.findNearest([0, 0.5, 0]);
        console.log('Nearest node:', nearest.concept_data.name);
        
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Run tests
runNavigatorTests();