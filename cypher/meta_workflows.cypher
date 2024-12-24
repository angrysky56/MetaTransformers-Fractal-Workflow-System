// Create a Meta-Workflow for Error Resolution
CREATE (meta_error_resolution:MetaWorkflow {
    name: "Error Resolution Meta-Workflow",
    description: "Meta-workflow for handling and resolving errors."
})

// Link the Error Handling Workflow to the Meta-Workflow
CREATE (meta_error_resolution_new:MetaWorkflow {
    name: "Error Resolution Meta-Workflow",
    description: "Meta-workflow for handling and resolving errors."
})

WITH meta_error_resolution
MATCH (wf_error:Workflow {name: "Error Handling Workflow"})
CREATE (meta_error_resolution)-[:INCLUDES]->(wf_error)
CREATE (meta_error_resolution)-[:INCLUDES]->(wf_error)

// Create a Meta-Workflow for Information Gathering
CREATE (meta_information_gathering:MetaWorkflow {
    name: "Information Gathering Meta-Workflow",
    description: "Meta-workflow for gathering information."
})

// Link the Web Search Workflow to the Information Gathering Meta-Workflow
WITH meta_information_gathering
MATCH (wf_search:Workflow {name: "Web Search Workflow"})
CREATE (meta_information_gathering)-[:INCLUDES]->(wf_search)

// Create Complex Task Meta-Workflow
CREATE (meta_complex:MetaWorkflow {
    name: "Complex Task Meta-Workflow",
    description: "Example of a complex meta-workflow"
})

// Link workflows in sequence within Complex Task Meta-Workflow
WITH *
MATCH (wf_search:Workflow {name: "Web Search Workflow"})
MATCH (wf_error:Workflow {name: "Error Handling Workflow"})
CREATE (meta_complex)-[:INCLUDES]->(wf_search)
CREATE (wf_search)-[:NEXT_IN_META]->(wf_error)
