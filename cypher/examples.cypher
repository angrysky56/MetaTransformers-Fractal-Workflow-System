// Example of a fractal workflow where "Process Data" includes "Clean Data" and "Analyze Data"
CREATE (wf_process_data:Workflow {name: "Process Data", description: "Workflow to process data."});
CREATE (step_process_start:Step {name: "Process Start", description: "Start data processing."});
CREATE (step_process_end:Step {name: "Process End", description: "End data processing."});
CREATE (wf_process_data)-[:STARTS_WITH]->(step_process_start);
CREATE (step_process_start)-[:NEXT]->(step_process_end);

CREATE (wf_clean_data:Workflow {name: "Clean Data", description: "Workflow to clean data."});
CREATE (wf_analyze_data:Workflow {name: "Analyze Data", description: "Workflow to analyze data."});

// Make "Process Data" include "Clean Data" and "Analyze Data"
MATCH (wf_process_data), (wf_clean_data), (wf_analyze_data)
CREATE (wf_process_data)-[:INCLUDES]->(wf_clean_data)
CREATE (wf_process_data)-[:INCLUDES]->(wf_analyze_data);

// Example of deeper nesting - "Clean Data" might include more specific cleaning steps
CREATE (wf_filter_data:Workflow {name: "Filter Data", description: "Workflow to filter data."});
MATCH (wf_clean_data), (wf_filter_data)
CREATE (wf_clean_data)-[:INCLUDES]->(wf_filter_data);

// Example of a workflow that includes itself (recursion) - be cautious with this in practice
// This is a conceptual example and needs careful handling in execution logic to prevent infinite loops
CREATE (wf_recursive_analysis:Workflow {name: "Recursive Analysis", description: "Workflow that recursively analyzes data."});
CREATE (step_recursive_start:Step {name: "Recursive Analysis Start", description: "Start recursive analysis."});
CREATE (step_recursive_check:Step {name: "Check Condition", description: "Check if the recursion condition is met."});
CREATE (step_recursive_analyze:Step {name: "Perform Analysis", description: "Perform the analysis step."});
CREATE (step_recursive_end:Step {name: "Recursive Analysis End", description: "End recursive analysis."});

CREATE (wf_recursive_analysis)-[:STARTS_WITH]->(step_recursive_start);
CREATE (step_recursive_start)-[:NEXT]->(step_recursive_check);
CREATE (step_recursive_check)-[:NEXT]->(step_recursive_analyze);
CREATE (step_recursive_analyze)-[:NEXT]->(step_recursive_end);

// Conceptually, make the workflow include itself based on a condition (this needs careful execution logic)
MATCH (wf_recursive_analysis), (step_recursive_check)
CREATE (step_recursive_check)-[:BRANCHES_TO {condition: "Needs further analysis"}]->(wf_recursive_analysis);
