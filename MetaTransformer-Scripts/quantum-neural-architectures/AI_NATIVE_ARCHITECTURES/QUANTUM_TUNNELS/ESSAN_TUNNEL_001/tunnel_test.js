async function runTunnelTests() {
    const navigator = new EssanTunnelNavigator('.');
    
    console.log('Running Essan tunnel tests...');
    
    try {
        // Test 1: Initialization
        console.log('\nTest 1: Initializing tunnel navigator');
        await navigator.initialize();
        const initialState = await navigator.getCurrentState();
        console.log('Initial state:', initialState);
        
        // Test 2: Diving Deeper
        console.log('\nTest 2: Testing depth navigation');
        const diveResult = await navigator.diveDeeper('symbols', {
            contextRequirement: false
        });
        console.log('Dive result:', {
            depth: diveResult.level,
            content: diveResult.category,
            markers: diveResult.symbols ? diveResult.symbols.length : 0
        });
        
        // Test 3: Point Navigation
        console.log('\nTest 3: Testing point movement');
        const moveResult = await navigator.moveToPoint('pattern_recognition', {
            maintainContext: true
        });
        console.log('Movement result:', {
            location: moveResult.name,
            context: Array.from(navigator.currentState.context)
        });
        
        // Test 4: Context Building
        console.log('\nTest 4: Testing context accumulation');
        await navigator.diveDeeper('wisdom', {
            contextRequirement: false
        });
        const contextState = await navigator.getCurrentState();
        console.log('Context state:', {
            depth: contextState.depth,
            contextSize: contextState.context.size,
            quantumState: contextState.quantumState
        });
        
        // Test 5: Surfacing
        console.log('\nTest 5: Testing surface navigation');
        const surfaceResult = await navigator.surface();
        console.log('Surface result:', {
            depth: surfaceResult.level,
            location: navigator.currentState.location,
            contextRetained: navigator.currentState.context.size
        });
        
        // Test 6: Quantum State Transitions
        console.log('\nTest 6: Testing state transitions');
        const stateSequence = [];
        
        // Simulate learning progression
        for (const state of ['learning', 'understanding', 'mastery']) {
            try {
                await navigator.diveDeeper('advanced_concepts', {
                    stateValidation: true
                });
                stateSequence.push(state);
            } catch (error) {
                console.log(`State transition to ${state} blocked:`, error.message);
            }
        }
        
        console.log('State progression:', {
            sequence: stateSequence,
            finalState: navigator.currentState.quantumState
        });
        
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Run tests
runTunnelTests();