async function runNavigatorTests() {
    const navigator = new TimeNavigator('.');
    
    console.log('Running time navigation tests...');
    
    try {
        // Test 1: Initialization
        console.log('\nTest 1: Initializing time navigator');
        await navigator.initialize();
        const initialState = await navigator.getCurrentState();
        console.log('Initial state:', {
            historySize: initialState.history_size,
            predictionsAvailable: initialState.predictions_available,
            currentTimestamp: initialState.current.timestamp
        });
        
        // Test 2: Forward Navigation
        console.log('\nTest 2: Testing forward navigation');
        const forwardResult = await navigator.moveForward(1, {
            predictIfNeeded: true,
            confidenceThreshold: 0.7
        });
        console.log('Forward navigation result:', {
            newState: forwardResult.state_id,
            timestamp: forwardResult.timestamp,
            metrics: forwardResult.projected_metrics || forwardResult.state_metrics
        });
        
        // Test 3: Backward Navigation
        console.log('\nTest 3: Testing backward navigation');
        const backwardResult = await navigator.moveBackward(1);
        console.log('Backward navigation result:', {
            newState: backwardResult.state_id,
            timestamp: backwardResult.timestamp,
            metrics: backwardResult.state_metrics
        });
        
        // Test 4: State Jump
        console.log('\nTest 4: Testing direct state jump');
        const jumpResult = await navigator.jumpToState('genesis', {
            validateTransition: true,
            preserveContext: true
        });
        console.log('Jump result:', {
            targetState: jumpResult.state_id,
            contextPreserved: jumpResult.active_contexts.length > 0
        });
        
        // Test 5: Path Analysis
        console.log('\nTest 5: Testing path analysis');
        const pathAnalysis = await navigator.analyzePath(
            'genesis',
            'projected_v2'
        );
        console.log('Path analysis:', {
            temporalDistance: pathAnalysis.temporal_distance,
            confidence: pathAnalysis.confidence,
            metricsEvolution: Object.keys(pathAnalysis.metrics_evolution)
        });
        
        // Test 6: Prediction Generation
        console.log('\nTest 6: Testing prediction generation');
        const prediction = await navigator.generatePrediction(2);
        console.log('Generated prediction:', {
            predictionId: prediction.state_id,
            confidence: prediction.confidence,
            predictedMetrics: prediction.projected_metrics
        });
        
        // Test 7: State Comparison
        console.log('\nTest 7: Testing state comparison');
        const currentState = await navigator.getCurrentState();
        const compareResult = navigator.analyzeStateChanges(
            currentState.current,
            prediction
        );
        console.log('State comparison:', {
            processChanges: {
                added: compareResult.process_changes.added.length,
                modified: compareResult.process_changes.modified.length,
                removed: compareResult.process_changes.removed.length
            },
            contextChanges: {
                added: compareResult.context_changes.added.length,
                modified: compareResult.context_changes.modified.length,
                removed: compareResult.context_changes.removed.length
            }
        });
        
        // Test 8: Metrics Evolution
        console.log('\nTest 8: Testing metrics evolution analysis');
        const metricsEvolution = navigator.analyzeMetricsEvolution(
            currentState.current,
            prediction
        );
        console.log('Metrics evolution:', {
            stabilityChange: metricsEvolution.stability?.change || 0,
            evolutionRateChange: metricsEvolution.evolution_rate?.change || 0,
            convergenceChange: metricsEvolution.convergence?.change || 0
        });
        
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Run tests
runNavigatorTests();