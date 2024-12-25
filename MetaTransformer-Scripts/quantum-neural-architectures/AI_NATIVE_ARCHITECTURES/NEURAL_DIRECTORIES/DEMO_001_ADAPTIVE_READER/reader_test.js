async function runReaderTests() {
    const reader = new AdaptiveReader('.');
    
    console.log('Running adaptive reader tests...');
    
    try {
        // Test 1: Initialization
        console.log('\nTest 1: Initializing reader');
        await reader.initialize();
        
        // Test 2: Basic Content Processing
        console.log('\nTest 2: Processing simple content');
        const result1 = await reader.processContent(
            "This is a test document about artificial intelligence and neural networks. " +
            "The system should be able to recognize and categorize technical content."
        );
        console.log('Processing result:', {
            categories: result1.categories,
            confidence: Object.values(result1.confidence)
                .reduce((sum, val) => sum + val, 0) / 
                Object.values(result1.confidence).length,
            activePath: result1.processingPath
        });
        
        // Test 3: Complex Content Processing
        console.log('\nTest 3: Processing complex content');
        const result2 = await reader.processContent(
            "The integration of quantum computing principles with neural network " +
            "architectures presents fascinating opportunities for advancing AI systems. " +
            "Recent developments in deep learning have shown promising results.",
            { learningRate: 0.2 }
        );
        console.log('Complex processing result:', {
            categories: result2.categories,
            metrics: result2.learningMetrics,
            activePath: result2.processingPath
        });
        
        // Test 4: Pattern Recognition
        console.log('\nTest 4: Testing pattern recognition');
        const testPatterns = await reader.extractPatterns(
            ['AI', 'Neural', 'Network', 'Learning', '2024', 'Deep'],
            { pattern_weights: { numeric: 1.2, proper: 1.0, simple: 0.8, complex: 1.1 } }
        );
        console.log('Pattern recognition:', {
            patternCount: testPatterns.length,
            types: testPatterns.map(p => p.type)
        });
        
        // Test 5: Learning Effects
        console.log('\nTest 5: Verifying learning effects');
        // Process same content twice to check learning
        const firstPass = await reader.processContent(
            "Neural networks and deep learning are transforming AI applications.",
            { learningRate: 0.15 }
        );
        
        const secondPass = await reader.processContent(
            "Neural networks and deep learning are transforming AI applications.",
            { learningRate: 0.15 }
        );

        console.log('Learning comparison:', {
            firstPassConfidence: Object.values(firstPass.confidence)
                .reduce((sum, val) => sum + val, 0) / 
                Object.values(firstPass.confidence).length,
            secondPassConfidence: Object.values(secondPass.confidence)
                .reduce((sum, val) => sum + val, 0) / 
                Object.values(secondPass.confidence).length,
            improvementMetrics: {
                categoriesFound: secondPass.learningMetrics.categoriesFound - 
                               firstPass.learningMetrics.categoriesFound,
                confidenceGain: secondPass.learningMetrics.averageConfidence - 
                               firstPass.learningMetrics.averageConfidence
            }
        });
        
        // Test 6: Feature Extraction
        console.log('\nTest 6: Testing feature extraction');
        const features = await reader.extractFeatures(
            "Testing advanced feature extraction capabilities with AI-related content."
        );
        console.log('Feature extraction:', {
            featureCount: features.length,
            samples: features.slice(0, 3)
        });
        
        // Test 7: Activation Path Analysis
        console.log('\nTest 7: Analyzing activation paths');
        const pathResult = await reader.processContent(
            "Complex neural architectures with deep learning capabilities.",
            { activationThreshold: 0.4 }
        );
        console.log('Activation path analysis:', {
            inputNodes: pathResult.processingPath.input.length,
            processingNodes: pathResult.processingPath.processing.length,
            outputNodes: pathResult.processingPath.output.length,
            metrics: pathResult.learningMetrics
        });
        
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Run tests
runReaderTests();