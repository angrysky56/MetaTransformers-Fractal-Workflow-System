// Test suite for FractalNavigator
async function runNavigationTests() {
    const navigator = new FractalNavigator('./concept_root');
    
    console.log('Running navigation tests...');
    
    try {
        // Test 1: Initial Context
        console.log('\nTest 1: Getting initial context');
        const initialContext = await navigator.getCurrentContext();
        console.log('Initial context:', initialContext);
        
        // Test 2: Dive into structure
        console.log('\nTest 2: Diving into structure branch');
        const structureNode = await navigator.dive('structure');
        console.log('Structure node:', structureNode);
        
        // Test 3: Traverse to behavior
        console.log('\nTest 3: Traversing to behavior branch');
        const behaviorNode = await navigator.traverse('behavior');
        console.log('Behavior node:', behaviorNode);
        
        // Test 4: Dive into learning
        console.log('\nTest 4: Diving into learning sub-branch');
        const learningNode = await navigator.dive('learning');
        console.log('Learning node:', learningNode);
        
        // Test 5: Surface back up
        console.log('\nTest 5: Surfacing up one level');
        const parentNode = await navigator.surface();
        console.log('Parent node:', parentNode);
        
        // Test 6: Get current path info
        console.log('\nTest 6: Getting path information');
        const pathInfo = await navigator.getPath();
        console.log('Path info:', pathInfo);
        
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Run tests
runNavigationTests();