# Neural Directory Demo: Adaptive Reader

## Overview
This demo implements a simple neural directory system that processes and categorizes text content, learning from its interactions.

## Structure
```
DEMO_001_ADAPTIVE_READER/
├── network_config.json       # Network configuration
├── input_layer/             # Content input nodes
│   └── node_001.json        # Text input processor
├── processing_layer/        # Analysis nodes
│   └── node_proc_001.json   # Content analyzer
└── output_layer/           # Categorization nodes
    └── node_output_001.json # Content categorizer
```

## How It Works

1. Input Layer
   - Receives text content
   - Activates based on content type
   - Routes to appropriate processors

2. Processing Layer
   - Analyzes content
   - Extracts patterns and keywords
   - Updates weights based on success

3. Output Layer
   - Categorizes processed content
   - Learns from feedback
   - Adapts categorization rules

## Usage Example
```python
# Initialize network
reader = AdaptiveReader("DEMO_001_ADAPTIVE_READER")

# Process content
result = reader.process_content("example.txt")

# Network learns from result
reader.update_weights(result.success_rate)
```

## Learning Mechanisms
- Weight adjustment based on success
- Pattern recognition improvement
- Category refinement
- Confidence thresholds

## Notes
- Simple demonstration of neural directory concept
- Shows adaptive behavior
- Includes basic learning mechanisms
- Demonstrates weight-based routing