async function runEvolutionTests() {
    const system = new EvolvingNeuralSystem('.');
    
    console.log('Running neural evolution tests...');
    
    try {
        // Test 1: Initialization
        console.log('\nTest 1: Initializing evolving neural system');
        await system.initialize();
        const initialState = await system.getCurrentState();
        console.log('Initial state:', initialState);
        
        // Test 2: Learning and Evolution
        console.log('\nTest 2: Testing learning and evolution');
        const learningResult = await system.learnAndEvolve({
            input: [1, 0, 1],
            target: [1, 1, 0]
        });
        console.log('Learning result:', {
            performance: learningResult.learningResult.performance,
            evolutionStage: learningResult.currentState.evolutionStage,
            predictionAvailable: !!learningResult.prediction
        });
        
        // Test 3: Evolution Review
        console.log('\nTest 3: Reviewing evolution');
        const review = await system.reviewEvolution({
            start: Date.now() - 3600000, // 1 hour ago
            end: Date.now()
        });
        console.log('Evolution review:', {
            patternCount: Object.keys(review.patterns).length,
            significantChanges: review.summary.significantChanges.length,
            overallProgress: review.summary.overallProgress
        });
        
        // Test 4: Pattern Analysis
        console.log('\nTest 4: Analyzing learning patterns');
        const patterns = await system.analyzeLearningPatterns({
            timestamps: [Date.now() - 1000, Date.now()],
            changes: [
                { layer1: 0.1, layer2: -0.2 },
                { layer1: 0.15, layer2: -0.25 }
            ],
            performance: [0.8, 0.85]
        });
        console.log('Pattern analysis:', {
            weightPatterns: patterns.weightEvolution.length,
            performancePatterns: patterns.performanceTrends.length,
            efficiencyPatterns: patterns.learningEfficiency.length
        });
        
        // Test 5: State Tracking
        console.log('\nTest 5: Checking current state');
        const currentState = await system.getCurrentState();
        console.log('Current system state:', {
            stage: currentState.evolutionStage,
            progress: currentState.learningProgress,
            timestamp: new Date(currentState.timestamp)
        });
        
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Run tests
runEvolutionTests();