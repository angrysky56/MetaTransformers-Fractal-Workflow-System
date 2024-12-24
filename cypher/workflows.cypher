// --- Foundational Error Handling Workflow ---
CREATE (wf_error:Workflow {name: "Error Handling Workflow", description: "A basic workflow for handling errors."});
CREATE (step_error_start:Step {name: "Error Start", description: "Initial step when an error occurs."});
CREATE (step_error_log:Step {name: "Log Error", description: "Log the details of the error."});
CREATE (step_error_notify:Step {name: "Notify Admin", description: "Notify the administrator about the error."});
CREATE (step_error_end:Step {name: "Error End", description: "Final step after error handling."});

CREATE (wf_error)-[:STARTS_WITH]->(step_error_start);
CREATE (step_error_start)-[:NEXT]->(step_error_log);
CREATE (step_error_log)-[:NEXT]->(step_error_notify);
CREATE (step_error_notify)-[:NEXT]->(step_error_end);

// --- Foundational Web Search Workflow ---
CREATE (wf_search:Workflow {name: "Web Search Workflow", description: "A simple workflow for performing a web search."});
CREATE (step_search_start:Step {name: "Search Start", description: "Initiate the web search process."});
CREATE (step_search_query:Step {name: "Formulate Query", description: "Formulate the search query."});
CREATE (step_search_execute:Step {name: "Execute Search", description: "Execute the search query."});
CREATE (step_search_analyze:Step {name: "Analyze Results", description: "Analyze the search results."});
CREATE (step_search_end:Step {name: "Search End", description: "Conclude the web search process."});

CREATE (wf_search)-[:STARTS_WITH]->(step_search_start);
CREATE (step_search_start)-[:NEXT]->(step_search_query);
CREATE (step_search_query)-[:NEXT]->(step_search_execute);
CREATE (step_search_execute)-[:NEXT]->(step_search_analyze);
CREATE (step_search_analyze)-[:NEXT]->(step_search_end);

// --- Example Single Step Workflow ---
CREATE (wf_greet:Workflow {name: "Greeting Workflow", description: "A simple workflow to greet the user."});
CREATE (step_greet:Step {name: "Greet User", description: "Display a greeting message."});
CREATE (wf_greet)-[:STARTS_WITH]->(step_greet);
